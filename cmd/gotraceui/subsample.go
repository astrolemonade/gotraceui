package main

// TODO notes: we benchmarked planar RLE vs packed DEFLATE. compression, RLE was slightly faster at worse compression.
// decompression, DEFLATE was 3x faster.

// TODO ahead of time generation of textures. when we request the texture for a zoom level, also generate the next zoom
// level in the background (depending on the direction in which the user is zooming.). Similar for panning to the left
// and right.

// TODO compress/delete unused textures

// TODO fix borders for hovered merged spans

// TODO document the architecture of this

// TODO update license/readme/... to attribute the dancing gopher

// TODO add back some tooltip to merged spans

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
	//
	// TODO(dh): the value of logOffset only matters because we iterate over all possible levels when looking for a
	// texture in O(levels). if we optimized that, we could set logOffset arbitrarily large, e.g. 64. Our data
	// structures are already sparse in the number of levels.
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
	stackPlaceholderUniform = image.NewUniform(colors[colorStatePlaceholderStackSpan].NRGBA())
	stackPlaceholderOp      = paint.NewImageOp(stackPlaceholderUniform)
	placeholderUniform      = stackPlaceholderUniform
	placeholderOp           = paint.NewImageOp(placeholderUniform)
)

type Texture struct {
	tex     *texture
	XScale  float32
	XOffset float32
}

type TextureStack struct {
	texs []Texture
}

func (tex TextureStack) Add(win *theme.Window, ops *op.Ops) (best bool) {
	for i, t := range tex.texs {
		_, imgOp, ok := t.tex.get(win)
		if !ok {
			continue
		}

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
		} else {
			imgOp.Add(ops)
		}

		// The offset only affects the clip, while the scale affects both the clip and the image.
		defer op.Affine(f32.Affine2D{}.Offset(f32.Pt(t.XOffset, 0))).Push(ops).Pop()
		// XXX is there a way we can fill the current clip, without having to specify its height?
		defer op.Affine(f32.Affine2D{}.Scale(f32.Pt(0, 0), f32.Pt(t.XScale, 40))).Push(ops).Pop()
		defer clip.Rect(image.Rect(0, 0, texWidth, 1)).Push(ops).Pop()

		paint.PaintOp{}.Add(ops)
		return i == 0
	}
	// We always expect a ready fallback texture to be available.
	panic("unreachable")
}

type texture struct {
	start trace.Timestamp
	// The texture's effective nsPerPx. When rendering a texture for a given nsPerPx, its XScale should be nsPerPx /
	// texture.nsPerPx.
	nsPerPx float64

	// Last frame this texture was used in
	lastUse uint64

	// The texture's zoom level, which is the log2 of the original nsPerPx, which may differ from the texture's
	// effective nsPerPx.
	level uint8

	data *theme.Future[textureData]
}

type textureData struct {
	computedIn time.Duration
	image      image.Image
	op         paint.ImageOp
}

func (tex *texture) get(win *theme.Window) (image.Image, paint.ImageOp, bool) {
	tex.lastUse = win.Frame
	return tex.getNoUse()
}

func (tex *texture) getNoUse() (image.Image, paint.ImageOp, bool) {
	data, ok := tex.data.ResultNoWait()
	if ok {
		return data.image, data.op, true
	} else {
		return nil, paint.ImageOp{}, false
	}
}

func (tex *texture) ready() bool {
	_, ok := tex.data.ResultNoWait()
	return ok
}

func (tex *texture) End() trace.Timestamp {
	return tex.start + trace.Timestamp(texWidth*tex.nsPerPx)
}

type Renderer struct {
	// textures is sorted by (logNsPerPx, start). In other words, it contains textures sorted by start time, grouped and
	// sorted by zoom level.
	//
	// Textures are aligned to multiples of the texture width * nsPerPx and are all of the same width, which is why we
	// can use a simple slice and binary search to find a matching texture, instead of needing an interval tree.
	//
	// OPT(dh): we could save memory by having one renderer per timeline instead of per track. For Sean's trace, we
	// could reduce usage from 118 MB to 9 MB (at a 10 GB bias.)
	textures []*texture

	// storage reused by Render
	texsOut []*texture
}

func texturesForLevelIndices(texs []*texture, level int) (start, end int) {
	start = sort.Search(len(texs), func(i int) bool {
		return texs[i].level >= uint8(level)
	})
	if start == len(texs) {
		return start, start
	}
	if texs[start].level != uint8(level) {
		return start, start
	}
	end = sort.Search(len(texs[start:]), func(i int) bool {
		return texs[i+start].level >= uint8(level)+1
	}) + start

	return start, end
}

func texturesForLevel(texs []*texture, level int) []*texture {
	start, end := texturesForLevelIndices(texs, level)
	return texs[start:end]
}

// MemoryUsage returns the approximate cumulative size in bytes of all cached textures. It doesn't account for slice
// headers or other overhead and only considers actual image data.
func (r *Renderer) MemoryUsage() uint64 {
	var size uint64
	for _, tex := range r.textures {
		img, _, _ := tex.getNoUse()
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

// renderTexture returns textures for the time range [start, start + texWidth * nsPerPx]. It may return multiple
// textures at different zoom levels, sorted by zoom level in descending order. That is, the first texture has the
// highest resolution.
func (r *Renderer) renderTexture(win *theme.Window, start trace.Timestamp, nsPerPx float64, spans Items[ptrace.Span], tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex, out []*texture) []*texture {
	if spans.Len() == 0 {
		return out
	}

	// The texture covers the time range [start, end]
	end := trace.Timestamp(math.Ceil(float64(start) + nsPerPx*texWidth))
	// The effective end of the texture is limited to the track's end. This is important in two places. 1, when
	// returning a uniform placeholder span, because it shouldn't be longer than the track. 2, when looking for an
	// existing, higher resolution texture that covers the entire range. It doesn't have to cover the larger void
	// produced by zooming out beyond the size of the track. It only has to cover up to the actual track end.
	end = min(end, spans.At(spans.Len()-1).End)

	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}

	var tex *texture
	logNsPerPx := int(math.Log2(nsPerPx)) + logOffset
	textures := texturesForLevel(r.textures, logNsPerPx)
	{
		// Find an exact match
		n := sort.Search(len(textures), func(i int) bool {
			return textures[i].start >= start
		})
		if n < len(textures) {
			c := textures[n]
			if c.start == start {
				tex = c
			}
		}
	}
	foundExact := tex != nil

	if tex != nil {
		out = append(out, tex)

		if tex.ready() {
			// The desired texture already exists and is ready, skip doing any further work.
			return out
		}
	}

	// Find a more zoomed in version of this texture, which we can downsample on the GPU by applying a smaller-than-one
	// scale factor.
	//
	// We'll only find such textures after looking at a whole track and then zooming out further.
	higherReady := false
	foundHigher := false
	for i := logNsPerPx - 1; i >= 0; i-- {
		textures := texturesForLevel(r.textures, i)
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
	for i := logNsPerPx + 1; i < 64+logOffset; i++ {
		textures := texturesForLevel(r.textures, i)
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
		placeholderTex := &texture{
			start:   start,
			nsPerPx: float64(end-start) / texWidth,
			level:   uint8(logNsPerPx),
			data: theme.Immediate(textureData{
				image: placeholderUniform,
				op:    placeholderOp,
			}),
		}
		out = append(out, placeholderTex)
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
		level:   uint8(logNsPerPx),
	}
	texsStart, texsEnd := texturesForLevelIndices(r.textures, logNsPerPx)
	n := sort.Search(len(r.textures[texsStart:texsEnd]), func(i int) bool {
		return r.textures[texsStart+i].start >= start
	})
	r.textures = slices.Insert(r.textures, texsStart+n, tex)
	out = slices.Insert(out, 0, tex)

	tex.data = theme.NewFuture[textureData](win, func(cancelled <-chan struct{}) textureData {
		if debugSlowRenderer {
			// Simulate a slow renderer.
			time.Sleep(time.Duration(rand.Intn(1000)+3000) * time.Millisecond)
		}
		return r.computeTexture(start, nsPerPx, spans, tex, tr, spanColor, cancelled)
	})
	return out
}

type pixel struct {
	sum       color.LinearSRGB
	sumWeight float64
}

// computeTexture computes a texture for the time range [start, start + texWidth * nsPerPx]. It will populate the
// appropriate fields in tex.
func (r *Renderer) computeTexture(start trace.Timestamp, nsPerPx float64, spans Items[ptrace.Span], tex *texture, tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex, cancelled <-chan struct{}) (out textureData) {
	debugTexturesComputing.Add(1)
	defer debugTexturesComputing.Add(-1)

	// The texture covers the time range [start, end]
	end := trace.Timestamp(math.Ceil(float64(start) + nsPerPx*texWidth))

	now := time.Now()
	defer func() {
		out.computedIn = time.Since(now)
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
		if i%10000 == 0 {
			select {
			case <-cancelled:
				return
			default:
			}
		}
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
		c := mappedColors[colorIdx]

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

	select {
	case <-cancelled:
		return
	default:
	}

	img := image.NewRGBA(image.Rect(0, 0, texWidth, 1))
	// Cache the conversion of the final pixel colors to sRGB.
	srgbCache := map[color.LinearSRGB]stdcolor.RGBA{}
	for x := range pixels {
		px := &pixels[x]
		px.sum.A = 1
		if px.sumWeight < 1 {
			// TODO(dh): can we ues transparent pixels instead?
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

	return textureData{
		image: img,
		op:    paint.NewImageOp(img),
	}
}

// Render returns a series of textures that when placed next to each other will display spans for the time range [start,
// end]. Each texture contains instructions for applying scaling and offsetting, and the series of textures may have
// gaps, expecting the background to already look as desired.
func (r *Renderer) Render(win *theme.Window, spans Items[ptrace.Span], nsPerPx float64, tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex, start trace.Timestamp, end trace.Timestamp, out []TextureStack) []TextureStack {
	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}
	if spanColor == nil {
		spanColor = defaultSpanColor
	}

	if spans.Len() == 0 {
		return out
	}

	if spans.At(0).State == statePlaceholder {
		// Don't go through the normal pipeline for making textures if the spans are placeholders (which happens when we
		// are dealing with compressed tracks.). Caching them would be wrong, as cached textures don't get invalidated
		// when spans change, and computing them is trivial.
		newStart := max(start, spans.At(0).Start)
		newEnd := min(end, spans.At(0).End)
		if newStart >= end || newEnd <= start {
			return nil
		}
		tex := &texture{
			start:   newStart,
			nsPerPx: float64(newEnd-newStart) / texWidth,
			data: theme.Immediate(textureData{
				image: stackPlaceholderUniform,
				op:    stackPlaceholderOp,
			}),
			level: uint8(math.Log2(nsPerPx)),
		}
		out = append(out, TextureStack{
			texs: []Texture{
				{
					tex:     tex,
					XScale:  float32(tex.nsPerPx / nsPerPx),
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
	//
	// Round nsPerPx to the next lower power of 2. A smaller nsPerPx will render a texture that is more zoomed in (and
	// thus detailed) than requested. This also means that the time range covered by the texture is smaller than
	// requested, and we may need to use multiple textures to cover the whole range.
	nsPerPx = math.Pow(2, math.Floor(math.Log2(nsPerPx)))
	m := texWidth * nsPerPx
	// Shift start point to the left to align with a multiple of texWidth * nsPerPx...
	start = trace.Timestamp(m * math.Floor(float64(start)/m))
	// ... but don't set start point to before the track's start.
	start = max(start, spans.At(0).Start)
	// Don't set end beyond track's end. This doesn't have any effect on the computed texture, but limits how many
	// textures we compute to cover the requested area.
	end = min(end, spans.At(spans.Len()-1).End)

	if start >= end {
		// We don't need a texture for an empty time interval
		return out[:0]
	}

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
		texs = r.renderTexture(win, start, nsPerPx, spans, tr, spanColor, texs[:0])

		var texs2 []Texture
		for _, tex := range texs {
			texs2 = append(texs2, Texture{
				tex: tex,
				// The texture has been rendered at a different scale than requested. At a minimum because we rounded it
				// down to a power of 2, but also because uniform colors modify the scale to limit the width they draw
				// at. Instruct the user to scale the texture to align things.
				XScale: float32(tex.nsPerPx / origNsPerPx),
				// The user wants to display the texture at origStart, but the texture's first pixel is actually for
				// tex.start (with tex.start <= start <= origStart). Instruct the user to offset the texture to align
				// things.
				XOffset: float32(float64(tex.start-origStart) / origNsPerPx),
			})
		}
		out = append(out, TextureStack{texs: texs2})
	}
	clear(texs)
	r.texsOut = texs[:0]

	return out
}
