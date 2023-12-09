package main

// TODO notes: we benchmarked planar RLE vs packed DEFLATE. compression, RLE was slightly faster at worse compression.
// decompression, DEFLATE was 3x faster.

// FIXME not too unexpectedly, we have seams when stitching textures, probably because of rounding errors and some
// samples not belonging to either texture.

/*
   After we mapped a texture request to a rounded start and nsPerPx, we try to find entries in the cache in the following order:
   - exact start + nsPerPx

   - [start, end] is superset of ours, nsPerPx as close to ours as possible. it can only be larger, not smaller, as we
     always use the same texture width and already round nsPerPx to powers of two. if the texture had a smaller nsPerPx,
     i.e. was more zoomed in, it would have to cover a smaller part of the trace given the same texture width. It is
     never worth computing this instead of an exact match, as computing more zoomed in textures is cheaper than
     computing zoomed out ones.

   - on-disk cache of the smallest nsPerPx required to view the entire trace. when super zoomed in this will be very low
     resolution, but a better choice than showing nothing. This is only useful if we can cheaply load it from disk when
     no better texture is available. Having to compute it would be slower than just computing the perfect texture.

   - a placeholder texture

*/

/*
   The UI requests textures to cover a [start, end] range of a track at a given nsPerPx.
   We round the nsPerPx to the next lower power of 2 and the start to the next lower multiple of nsPerPx * texture width.

   The resulting textures will be slightly more zoomed in than what the UI requested. The UI will scale the texture by a factor <=1
   to zoom out as needed. This has the benefit that when the user is zooming in continuously, we can downsample the
   texture on the GPU side, instead of having to upsample.

   We may generate more than one texture to span the whole UI width, depending on the UI's width in relation to the
   fixed texture width and panning. We want to find a good balance for texture size. Too narrow and we have to send more
   commands to the GPU and handle more entries in the cache. Too wide and we increase latency when having to compute a
   new texture.
*/

// TODO ahead of time generation of textures. when we request the texture for a zoom level, also generate the next zoom
// level in the background (depending on the direction in which the user is zooming.). Similar for panning to the left
// and right.

// TODO when animating a zoom, we may animate to a new level before we ever had a chance to finish computing the
// previous level. in that case, it would make sense to cancel the computation of the previous level. More generally, we
// could cancel every texture that wasn't used in a frame.

import (
	"fmt"
	"image"
	stdcolor "image/color"
	"math"
	"math/rand"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"honnef.co/go/gotraceui/clip"
	"honnef.co/go/gotraceui/color"
	"honnef.co/go/gotraceui/theme"
	"honnef.co/go/gotraceui/trace"
	"honnef.co/go/gotraceui/trace/ptrace"

	"gioui.org/f32"
	"gioui.org/op"
	"gioui.org/op/paint"
)

const (
	// Simulate a slow system by delaying texture computation by a random amount of time.
	debugSlowRenderer     = false
	debugDisplayGradients = false
	debugDisplayZoom      = false

	texWidth = 8192
	// Offset log2(nsPerPx) by this much to ensure it is positive. 16 allows for 1 ns / 64k pixels, which realistically
	// can never be used, because Gio doesn't support clip areas larger than 8k x 8k, and we don't allow zooming out
	// more than 1 / window_width.
	logOffset = 16
)

// Counter of currently computing textures.
var debugTexturesComputing atomic.Int64

var pixelsPool = &sync.Pool{
	New: func() any {
		s := make([]pixel, texWidth)
		return &s
	},
}

var (
	backgroundUniform = image.NewUniform(stdcolor.NRGBA{0xFF, 0xFF, 0xEB, 0xFF})
	backgroundOp      = paint.NewImageOp(backgroundUniform)
	// XXX find something better to display when we have no texture
	placeholderUniform      = image.NewUniform(stdcolor.NRGBA{0x00, 0xFF, 0x00, 0xFF})
	placeholderOp           = paint.NewImageOp(placeholderUniform)
	stackPlaceholderUniform = image.NewUniform(colors[colorStatePlaceholderStackSpan].NRGBA())
	stackPlaceholderOp      = paint.NewImageOp(stackPlaceholderUniform)
)

type textureKey struct {
	start   trace.Timestamp
	nsPerPx float64
}

type Texture struct {
	tex     *texture
	XScale  float32
	XOffset float32
}

type TextureStack struct {
	texs []Texture
}

func (tex TextureStack) Add(ops *op.Ops) {
	for _, t := range tex.texs {
		t.tex.mu.RLock()
		if t.tex.image != nil {
			t.tex.op.Add(ops)
			t.tex.mu.RUnlock()

			if debugDisplayGradients {
				// Display gradients in place of the texture. Good for seeing texture boundaries.
				paint.LinearGradientOp{Stop1: f32.Pt(0, 0), Stop2: f32.Pt(texWidth, 0), Color1: stdcolor.NRGBA{0xFF, 0x00, 0x00, 0xFF}, Color2: stdcolor.NRGBA{0x00, 0x00, 0xFF, 0xFF}}.Add(ops)
			} else if debugDisplayZoom {
				// Display the zoom level of the texture.
				ff := 128.0 * t.XScale
				var f byte
				if ff < 0 {
					f = 0
				} else if ff > 255 {
					f = 255
				} else {
					f = byte(ff)
				}
				paint.ColorOp{Color: stdcolor.NRGBA{f, f, f, 0xFF}}.Add(ops)
			}

			// The offset only affects the clip, while the scale affects both the clip and the image.
			defer op.Affine(f32.Affine2D{}.Offset(f32.Pt(t.XOffset, 0))).Push(ops).Pop()
			// XXX is there a way we can fill the current clip, without having to specify its height?
			defer op.Affine(f32.Affine2D{}.Scale(f32.Pt(0, 0), f32.Pt(t.XScale, 40))).Push(ops).Pop()
			defer clip.Rect(image.Rect(0, 0, texWidth, 1)).Push(ops).Pop()

			paint.PaintOp{}.Add(ops)
			return
		}
		t.tex.mu.RUnlock()
	}
}

type texture struct {
	start   trace.Timestamp
	nsPerPx float64

	mu         sync.RWMutex
	computedIn time.Duration
	image      image.Image
	op         paint.ImageOp
}

func (tex *texture) ready() bool {
	tex.mu.RLock()
	defer tex.mu.RUnlock()
	return tex.image != nil
}

func (tex *texture) End() trace.Timestamp {
	return tex.start + trace.Timestamp(texWidth*tex.nsPerPx)
}

type Renderer struct {
	exactTextures map[textureKey]*texture
	// textures is indexed by log2(nsPerPx). We allow for 80 levels [-16, 64]; 2**64 ns is 585 years and there will never be a
	// valid reason for zooming in or out that far.
	//
	// Textures are aligned to multiples of the texture width * nsPerPx and are all of teh same width, which is why we
	// can use a simple slice and binary search, instead of needing an interval tree.
	//
	// OPT(dh): this is a lot of memory to spend per track
	textures [64 + logOffset][]*texture

	// OPT(dh): can we afford this once per track? and should we? this mapping is identical for all tracks.
	mappedColors [len(colors)]color.LinearSRGB

	// storage reused by Render
	texsOut []*texture
}

// MemoryUsage returns the approximate cumulative size in bytes of all cached textures. It doesn't account for slice
// headers or other overhead and only considers actual image data.
func (r *Renderer) MemoryUsage() uint64 {
	var size uint64
	for _, tex := range r.exactTextures {
		tex.mu.RLock()
		img := tex.image
		tex.mu.RUnlock()
		switch img := img.(type) {
		case *image.Uniform:
			size += 4
		case *image.RGBA:
			size += uint64(len(img.Pix))
		case nil:
		default:
			panic(fmt.Sprintf("unhandled type %T", img))
		}
	}
	return size
}

func NewRenderer() *Renderer {
	r := &Renderer{
		exactTextures: map[textureKey]*texture{},
	}
	for i, c := range colors {
		r.mappedColors[i] = c.MapToSRGBGamut()
	}

	return r
}

// OPT reuse slice storage
func (r *Renderer) renderTexture(start trace.Timestamp, nsPerPx float64, spans Items[ptrace.Span], tr *Trace, spanColor func(Items[ptrace.Span], *Trace) colorIndex, out []*texture) []*texture {
	// OPT(dh): reuse slice
	// XXX this lookup doesn't allow using multiple textures to stitch together a bigger one. that is, when we need a texture for [0, 32] then we can't use [0, 16] + [16, 32]
	start = max(start, 0)
	end := trace.Timestamp(math.Ceil(float64(start) + nsPerPx*texWidth))
	end = min(end, tr.End())

	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}

	texKey := textureKey{
		start:   start,
		nsPerPx: nsPerPx,
	}
	tex := r.exactTextures[texKey]
	foundExact := tex != nil

	if tex != nil {
		out = append(out, tex)

		if tex.ready() {
			// The desired texture already exists and is ready, skip doing any further work.
			return out
		}
	}

	// XXX this can't possibly be efficient
	//
	// Find a more zoomed in version of this texture, which we can downsample on the GPU by applying a smaller-than-one
	// scale factor.
	//
	// We'll only find such textures after looking at a whole track and then zooming out further.
	higherReady := false
	foundHigher := false
	logNsPerPx := int(math.Log2(nsPerPx)) + logOffset
	for i := logNsPerPx - 1; i >= 0; i-- {
		textures := r.textures[i]
		n := sort.Search(len(textures), func(j int) bool {
			return r.textures[i][j].End() >= end
		})
		if n == len(textures) {
			continue
		}
		f := textures[n]
		if f.start <= start {
			out = append(out, f)
			if f.ready() {
				// Don't collect more textures that'll never be used.
				higherReady = true
				break
			}
		}
	}

	if higherReady {
		// A higher resolution texture is already ready. There is no point in looking for lower resolution ones, or
		// computing an exact match.
		return out
	}

	// Find a less zoomed in texture that covers our time range. We can upsample it on the GPU to get a blurry preview
	// of the final texture.
	for i := logNsPerPx + 1; i < len(r.textures); i++ {
		textures := r.textures[i]
		n := sort.Search(len(textures), func(j int) bool {
			return textures[j].End() >= end
		})
		if n == len(textures) {
			continue
		}
		f := textures[n]
		if f.start <= start {
			out = append(out, f)
			if f.ready() {
				// Don't collect more textures that'll never be used.
				break
			}
		}
	}

	{
		greenTex := &texture{
			start:   start,
			nsPerPx: nsPerPx,
			image:   placeholderUniform,
			op:      placeholderOp,
		}
		out = append(out, greenTex)
	}

	if foundHigher || foundExact {
		// We're already waiting on either the exact match, or a higher resolution texture. In neither case do we want
		// to start computing the exact match (again.)
		return out
	}

	if tex != nil {
		panic("unreachable")
	}

	tex = &texture{
		start:   start,
		nsPerPx: nsPerPx,
	}
	r.exactTextures[texKey] = tex
	n := sort.Search(len(r.textures[logNsPerPx]), func(i int) bool {
		return r.textures[logNsPerPx][i].start >= start
	})
	r.textures[logNsPerPx] = slices.Insert(r.textures[logNsPerPx], n, tex)
	out = slices.Insert(out, 0, tex)

	go r.computeTexture(start, end, nsPerPx, spans, tex, tr, spanColor)
	return out
}

type pixel struct {
	sum       color.LinearSRGB
	sumWeight float64
}

func (r *Renderer) computeTexture(start, end trace.Timestamp, nsPerPx float64, spans Items[ptrace.Span], tex *texture, tr *Trace, spanColor func(Items[ptrace.Span], *Trace) colorIndex) {
	debugTexturesComputing.Add(1)
	defer debugTexturesComputing.Add(-1)

	now := time.Now()
	defer func() {
		tex.mu.Lock()
		tex.computedIn = time.Since(now)
		tex.mu.Unlock()
	}()

	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}

	if start >= end {
		tex.mu.Lock()
		defer tex.mu.Unlock()
		tex.image = backgroundUniform
		tex.op = backgroundOp
		return
	}

	first := sort.Search(spans.Len(), func(i int) bool {
		return spans.At(i).End >= start
	})
	last := sort.Search(spans.Len(), func(i int) bool {
		return spans.At(i).Start >= end
	})
	if first >= last {
		tex.mu.Lock()
		defer tex.mu.Unlock()
		tex.image = backgroundUniform
		tex.op = backgroundOp
		return
	}

	if debugSlowRenderer {
		// Simulate a slow renderer.
		time.Sleep(time.Duration(rand.Intn(1000)+3000) * time.Millisecond)
	}

	pixelsPtr := pixelsPool.Get().(*[]pixel)
	pixels := *pixelsPtr
	clear(*pixelsPtr)
	defer pixelsPool.Put(pixelsPtr)

	addSample := func(bin int, w float64, v color.LinearSRGB) {
		if w == 0 {
			return
		}
		if bin >= len(pixels) {
			// XXX
			return
		}
		if bin < 0 {
			// XXX
			return
		}

		if w == 1 {
			pixels[bin] = pixel{
				sum:       v,
				sumWeight: 1,
			}
			return
		}
		px := &pixels[bin]
		if px.sumWeight+w > 1 {
			// Adjust for rounding errors
			w = 1 - px.sumWeight
		}
		px.sum.R += v.R * float32(w)
		px.sum.G += v.G * float32(w)
		px.sum.B += v.B * float32(w)
		px.sumWeight += w
	}

	for i := first; i < last; i++ {
		span := spans.At(i)

		firstBucket := float64(span.Start-start) / nsPerPx
		lastBucket := float64(span.End-start) / nsPerPx

		if firstBucket >= texWidth {
			break
		}
		if lastBucket < 0 {
			continue
		}

		firstBucket = max(firstBucket, 0)
		lastBucket = min(lastBucket, texWidth)

		colorIdx := spanColor(spans.Slice(i, i+1), tr)
		c := r.mappedColors[colorIdx]

		if int(firstBucket) == int(lastBucket) {
			// falls into a single bucket
			w := float64(span.Duration()) / nsPerPx
			addSample(int(firstBucket), w, c)
		} else {
			// falls into at least two buckets

			_, frac := math.Modf(firstBucket)
			w1 := 1 - frac
			_, frac = math.Modf(lastBucket)
			w2 := frac

			addSample(int(firstBucket), w1, c)
			addSample(int(lastBucket), w2, c)
		}

		for i := int(firstBucket) + 1; i < int(lastBucket); i++ {
			// All the full buckets between the first and last one
			addSample(i, 1, c)
		}
	}

	img := image.NewRGBA(image.Rect(0, 0, texWidth, 1))
	// Cache the conversion of the final pixel colors to sRGB.
	srgbCache := map[color.LinearSRGB]stdcolor.RGBA{}
	for x := range pixels {
		px := &pixels[x]
		px.sum.A = 1
		if px.sumWeight < 1 {
			w := 1 - px.sumWeight
			px.sum.R += float32(w)
			px.sum.G += float32(w)
			px.sum.B += float32((212.0 / 255.0) * w)
			px.sumWeight = 1
		}

		srgb, ok := srgbCache[px.sum]
		if !ok {
			srgb = stdcolor.RGBAModel.Convert(px.sum.SRGB()).(stdcolor.RGBA)
			srgbCache[px.sum] = srgb
		}
		i := img.PixOffset(x, 0)
		s := img.Pix[i : i+4 : i+4] // Small cap improves performance, see https://golang.org/issue/27857
		s[0] = srgb.R
		s[1] = srgb.G
		s[2] = srgb.B
		s[3] = srgb.A
	}

	tex.mu.Lock()
	defer tex.mu.Unlock()
	tex.op = paint.NewImageOp(img)
	tex.image = img
}

func (r *Renderer) Render(win *theme.Window, spans Items[ptrace.Span], nsPerPx float64, tr *Trace, spanColor func(Items[ptrace.Span], *Trace) colorIndex, start trace.Timestamp, end trace.Timestamp, out []TextureStack) []TextureStack {
	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}
	if spanColor == nil {
		spanColor = defaultSpanColor
	}

	if spans.Len() > 0 && spans.At(0).State == statePlaceholder {
		// Don't go through the normal pipeline for making textures if the spans are placeholders (which happens when we
		// are dealing with compressed tracks.). Caching them would be wrong, as cached textures don't get invalidated
		// when spans change, and computing them is trivial.
		newStart := max(start, spans.At(0).Start)
		newEnd := min(end, spans.At(0).End)
		if newStart >= end || newEnd <= start {
			return nil
		}
		out = append(out, TextureStack{
			texs: []Texture{
				Texture{
					tex: &texture{
						start:   newStart,
						nsPerPx: nsPerPx,
						image:   stackPlaceholderUniform,
						op:      stackPlaceholderOp,
					},
					XScale:  float32(float64(newEnd-newStart) / (nsPerPx * texWidth)),
					XOffset: float32(float64(newStart-start) / nsPerPx),
				},
			},
		})
		return out
	}

	origStart := start
	origNsPerPx := nsPerPx
	// TODO does log2(nsPerPx) grow the way we expect? in particular, the more zoomed out we are, the longer we want to
	// spend scaling the same zoom level.
	nsPerPx = math.Pow(2, math.Floor(math.Log2(nsPerPx)))
	m := texWidth * nsPerPx
	start = trace.Timestamp(m * math.Floor(float64(start)/m))

	// texWidth is an integer pixel amount, and nsPerPx is rounded to a power of 2, ergo also integer.
	step := trace.Timestamp(texWidth * nsPerPx)
	if step == 0 {
		// For a texWidth, if the user zooms to log2(pxPerNs) < -log2(texWidth), step will truncate to zero. As long as
		// texWidth >= 8192, this should be impossible, because Gio doesn't support clip areas larger than 8192, and
		// nsPerPx has to be at least 1 / window_width.
		if texWidth < 8192 {
			panic("got zero step with a texWidth < 8192")
		} else {
			panic("got zero step despite a texWidth >= 8192")
		}
	}
	texs := r.texsOut
	for start := start; start < end; start += step {
		texs = r.renderTexture(start, nsPerPx, spans, tr, spanColor, texs[:0])

		var texs2 []Texture
		for _, tex := range texs {
			texs2 = append(texs2, Texture{
				tex:     tex,
				XScale:  float32(tex.nsPerPx / origNsPerPx),
				XOffset: float32(float64(tex.start-origStart) / origNsPerPx),
			})
		}
		out = append(out, TextureStack{texs: texs2})
	}
	clear(texs)
	r.texsOut = texs[:0]

	return out
}
