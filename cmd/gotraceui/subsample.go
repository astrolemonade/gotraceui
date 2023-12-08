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
	"honnef.co/go/gotraceui/container"
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

type textureKey struct {
	Start   trace.Timestamp
	NsPerPx float64
}

type DisplayTexture2 struct {
	tex     *texture
	XScale  float32
	XOffset float32
}

type DisplayTexture struct {
	texs []DisplayTexture2
}

func (tex DisplayTexture) Add(ops *op.Ops) {
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

			defer op.Affine(f32.Affine2D{}.Offset(f32.Pt(t.XOffset, 0))).Push(ops).Pop()
			defer op.Affine(f32.Affine2D{}.Scale(f32.Pt(0, 0), f32.Pt(t.XScale, 40))).Push(ops).Pop()
			defer clip.Rect(image.Rect(0, 0, texWidth, 40)).Push(ops).Pop()
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
	// OPT(dh): this is a lot of memory to spend per track
	textures [64 + logOffset]*container.IntervalTree[trace.Timestamp, *texture]

	// OPT(dh): can we afford this once per track?
	mappedColors [len(colors)]color.LinearSRGB
}

func NewRenderer() *Renderer {
	r := &Renderer{
		exactTextures: map[textureKey]*texture{},
	}
	for i := range r.textures {
		r.textures[i] = container.NewIntervalTree[trace.Timestamp, *texture]()
	}
	for i, c := range colors {
		r.mappedColors[i] = c.MapToSRGBGamut()
	}

	return r
}

// OPT reuse slice storage
func (r *Renderer) renderTexture(start trace.Timestamp, nsPerPx float64, spans Items[ptrace.Span], tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex) []*texture {
	// OPT(dh): reuse slice
	// XXX this lookup doesn't allow using multiple textures to stitch together a bigger one. that is, when we need a texture for [0, 32] then we can't use [0, 16] + [16, 32]
	start = max(start, 0)
	end := trace.Timestamp(math.Ceil(float64(start) + nsPerPx*texWidth))
	end = min(end, tr.End())

	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}

	texKey := textureKey{
		Start:   start,
		NsPerPx: nsPerPx,
	}
	tex := r.exactTextures[texKey]
	foundExact := tex != nil

	var out []*texture
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
		// OPT(dh): reuse slice
		found := r.textures[i].Find(start, end, nil)
		for _, f := range found {
			// XXX instead of rejecting items here, run the correct query on the interval tree. right now we're selecting
			// all nodes whose start lie in [start, end], when really we want all nodes whose [start, end] are a subset of
			// the query range.
			if f.Value.Value.start > start || f.Value.Value.End() < end {
				continue
			}

			out = append(out, f.Value.Value)
			foundHigher = true
			if f.Value.Value.ready() {
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

	// XXX this can't possibly be efficient
	//
	// Find a less zoomed in texture that covers our time range. We can upsample it on the GPU to get a blurry preview
	// of the final texture.
	for i := logNsPerPx + 1; i < len(r.textures); i++ {
		// OPT(dh): reuse slice
		found := r.textures[i].Find(start, end, nil)
		for _, f := range found {
			fmt.Println(i, f.Value.Value.start, f.Value.Value.End())
			// XXX instead of rejecting items here, run the correct query on the interval tree. right now we're selecting
			// all nodes whose start lie in [start, end], when really we want all nodes whose [start, end] are a subset of
			// the query range.
			if f.Value.Value.start > start || f.Value.Value.End() < end {
				continue
			}

			out = append(out, f.Value.Value)
			if f.Value.Value.ready() {
				// Don't collect more textures that'll never be used.
				break
			}
		}
	}

	{
		// XXX find something better to display when we have no texture
		green := image.NewUniform(stdcolor.NRGBA{0x00, 0xFF, 0x00, 0xFF})
		greenTex := &texture{
			start:   start,
			nsPerPx: nsPerPx,
			image:   green,
			op:      paint.NewImageOp(green),
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
	r.textures[logNsPerPx].Insert(start, end, tex)
	slices.Insert(out, 0, tex)

	go r.computeTexture(start, end, nsPerPx, spans, tex, tr, spanColor)
	return out
}

type pixel struct {
	sum       color.LinearSRGB
	sumWeight float64
}

var pixelsPool = &sync.Pool{
	New: func() any {
		s := make([]pixel, texWidth)
		return &s
	},
}

func (r *Renderer) computeTexture(start, end trace.Timestamp, nsPerPx float64, spans Items[ptrace.Span], tex *texture, tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex) {
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
	first := sort.Search(spans.Len(), func(i int) bool {
		return spans.At(i).End >= start
	})
	last := sort.Search(spans.Len(), func(i int) bool {
		return spans.At(i).Start >= end
	})
	if last <= first {
		img := image.NewUniform(stdcolor.NRGBA{0xFF, 0xFF, 0xEB, 0xFF})
		tex.mu.Lock()
		defer tex.mu.Unlock()
		tex.image = img
		tex.op = paint.NewImageOp(img)
		// OPT(dh): taking 1 us to get here seems a bit long. is it because of the binary search?
		return
	}

	if debugSlowRenderer {
		// Simulate a slow renderer.
		time.Sleep(time.Duration(rand.Intn(1000)) * time.Millisecond)
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

		colorIdx := spanColor(spans.At(i), tr)
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

func (r *Renderer) Render(win *theme.Window, spans Items[ptrace.Span], nsPerPx float64, tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex, start trace.Timestamp, end trace.Timestamp, out []DisplayTexture) []DisplayTexture {
	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}
	if spanColor == nil {
		spanColor = defaultSpanColor
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
	for start := start; start < end; start += step {
		texs := r.renderTexture(start, nsPerPx, spans, tr, spanColor)

		var texs2 []DisplayTexture2
		for _, tex := range texs {
			texs2 = append(texs2, DisplayTexture2{
				tex:     tex,
				XScale:  float32(tex.nsPerPx / origNsPerPx),
				XOffset: float32(float64(tex.start-origStart) / origNsPerPx),
			})
		}
		out = append(out, DisplayTexture{texs: texs2})
	}

	return out
}
