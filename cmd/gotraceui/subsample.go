package main

// OPT cancel textures

// TODO ahead of time generation of textures. when we request the texture for a zoom level, also generate the next zoom
// level in the background (depending on the direction in which the user is zooming.). Similar for panning to the left
// and right.

// TODO fix borders for hovered merged spans

// TODO document the architecture of this

// TODO update license/readme/... to attribute the dancing gopher

// TODO add back some tooltip to merged spans

import (
	"bytes"
	"compress/flate"
	"fmt"
	"image"
	stdcolor "image/color"
	"io"
	"log"
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
	"honnef.co/go/gotraceui/mysync"
	"honnef.co/go/gotraceui/theme"
	"honnef.co/go/gotraceui/trace"
	"honnef.co/go/gotraceui/trace/ptrace"

	"gioui.org/f32"
	"gioui.org/op"
	"gioui.org/op/paint"
)

const (
	// Simulate a slow system by delaying texture computation by a random amount of time.
	debugSlowRenderer      = false
	debugDisplayGradients  = false
	debugDisplayZoom       = false
	debugTraceRenderer     = false
	debugTextureCompaction = false

	texWidth = 8192
	// Offset log2(nsPerPx) by this much to ensure it is positive. 16 allows for 1 ns / 64k pixels, which realistically
	// can never be used, because Gio doesn't support clip areas larger than 8k x 8k, and we don't allow zooming out
	// more than 1 / window_width.
	//
	// TODO(dh): the value of logOffset only matters because we iterate over all possible levels when looking for a
	// texture in O(levels). if we optimized that, we could set logOffset arbitrarily large, e.g. 64. Our data
	// structures are already sparse in the number of levels.
	logOffset = 16

	// How many frames to wait between compaction attempts
	compactInterval = 100
	// Maximum memory to spend on storing textures
	maxTextureMemoryUsage = 100 * 1024 * 1024
	// 10% for compressed textures
	maxCompressedMemoryUsage = maxTextureMemoryUsage / 10
	// the remainder for decompressed textures
	maxRGBAMemoryUsage = maxTextureMemoryUsage - maxCompressedMemoryUsage
)

// Counter of currently computing textures.
var debugTexturesComputing atomic.Int64

var pixelsPool = &sync.Pool{
	New: func() any {
		s := make([]pixel, texWidth)
		return &s
	},
}

var flateWriterPool = &sync.Pool{
	New: func() any {
		return (*flate.Writer)(nil)
	},
}

var closedDoneChannel = makeClosedChan[struct{}]()

var (
	stackPlaceholderUniform = image.NewUniform(colors[colorStatePlaceholderStackSpan].NRGBA())
	stackPlaceholderOp      = paint.NewImageOp(stackPlaceholderUniform)
	placeholderUniform      = stackPlaceholderUniform
	placeholderOp           = paint.NewImageOp(placeholderUniform)
)

func renderTrace(f string, v ...any) {
	if debugTraceRenderer {
		log.Printf(f, v...)
	}
}

type Texture struct {
	tex     *texture
	XScale  float32
	XOffset float32
}

type TextureStack struct {
	texs []Texture
}

func (tex TextureStack) Add(win *theme.Window, tm *TextureManager, tr *Trace, ops *op.Ops) (best bool) {
	for i, t := range tex.texs {
		_, imgOp, ok := tm.Image(win, tr, t.tex)
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
	panic("unreachable")
}

type texture struct {
	track *Track
	// We cannot use track.spans because that might change between planning and computing textures. For example, CPU
	// sampling tracks get niled out when the widget scrolls out of view. We don't want to handle nil spans and
	// not-yet-ready spans and risk restarting futures in several places.
	spans Items[ptrace.Span]
	start trace.Timestamp
	// The texture's effective nsPerPx. When rendering a texture for a given nsPerPx, its XScale should be nsPerPx /
	// texture.nsPerPx.
	nsPerPx float64

	// Last frame this texture was used in
	lastUse uint64

	computed textureComputed
	realized textureRealized

	// The texture's zoom level, which is the log2 of the original nsPerPx, which may differ from the texture's
	// effective nsPerPx.
	level uint8

	tracked bool
	// This is an ephemeral texture that isn't tracked in any caches, returned only to hold a uniform.
	ephemeral bool
}

type textureComputed struct {
	done       chan struct{}
	computedIn time.Duration
	// Compressed stores the compressed version of the texture. This is always populated for image.RGBA-backed
	// textures once computing it has finished. That is, we don't compress textures on demand. The average
	// compression ratio depends on the shape of the trace and the zoom level, but seems to be anywhere around
	// 15x to 150x. For 8192-wide textures, the increase from 32 KiB to 34 KiB (at a 15x ratio) is only a
	// 6.25% increase in size. In return for that increase we don't have to spend 500 µs per texture we want
	// to compress, and we only have to compress a given texture once.
	compressed []byte
	uniform    *image.Uniform
}

type textureRealized struct {
	done  chan struct{}
	image image.Image
	op    paint.ImageOp
}

func (tex *texture) ready() bool {
	if tex.realized.done != nil {
		return CanRecv(tex.realized.done)
	}
	return CanRecv(tex.computed.done) && tex.computed.uniform != nil
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

type RendererStatistics struct {
	RealizedUniforms atomic.Uint64
	RealizedRGBAs    atomic.Uint64
	CompressedNum    atomic.Uint64
	CompressedSize   atomic.Uint64
}

func (stats *RendererStatistics) String() string {
	var (
		uniforms       = stats.RealizedUniforms.Load()
		rgbas          = stats.RealizedRGBAs.Load()
		compressedNum  = stats.CompressedNum.Load()
		compressedSize = stats.CompressedSize.Load()
	)
	f := `Realized uniforms: %d (%f MiB)
Realized RGBAs: %d (%f MiB)
Compressed RGBAs: %d (%f MiB, %f MiB uncompressed)`
	return fmt.Sprintf(
		f,
		uniforms, float64(uniforms*4)/1024/1024,
		rgbas, float64(rgbas*texWidth*4)/1024/1024,
		compressedNum, float64(compressedSize)/1024/1024, float64(stats.CompressedNum.Load()*texWidth*4)/1024/1024,
	)
}

// planTextures returns textures for the time range [start, start + texWidth * nsPerPx]. It may return multiple
// textures at different zoom levels, sorted by zoom level in descending order. That is, the first texture has the
// highest resolution.
func (r *Renderer) planTextures(
	win *theme.Window,
	track *Track,
	spans Items[ptrace.Span],
	start trace.Timestamp,
	nsPerPx float64,
	out []*texture,
) []*texture {
	renderTrace("  planning texture at %d ns @ %f ns/px for track in %q", start, nsPerPx, track.parent.shortName)

	if spans.Len() == 0 {
		renderTrace("    no spans to plan")
		return out
	}

	// The texture covers the time range [start, end]
	end := trace.Timestamp(math.Ceil(float64(start) + nsPerPx*texWidth))
	// The effective end of the texture is limited to the track's end. This is important in two places. 1, when
	// returning a uniform placeholder span, because it shouldn't be longer than the track. 2, when looking for an
	// existing, higher resolution texture that covers the entire range. It doesn't have to cover the larger void
	// produced by zooming out beyond the size of the track. It only has to cover up to the actual track end.
	end = min(end, spans.At(spans.Len()-1).End)

	renderTrace("    effective end is %d", end)

	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}

	var tex *texture
	logNsPerPx := int(math.Log2(nsPerPx)) + logOffset
	{
		// Find an exact match
		textures := texturesForLevel(r.textures, logNsPerPx)
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
	// XXX add method that encapsulates tex.computed.done != nil
	foundExact := tex != nil && tex.computed.done != nil

	if tex != nil && (tex.computed.done != nil) {
		out = append(out, tex)

		if tex.ready() {
			// The desired texture already exists and is ready, skip doing any further work.
			renderTrace("    exact match is already ready to use, returning early")
			return out
		} else {
			renderTrace("    found exact match, but it isn't ready yet")
		}
	}

	// Find a more zoomed in version of this texture, which we can downsample on the GPU by applying a smaller-than-one
	// scale factor.
	//
	// We'll only find such textures after looking at a whole track and then zooming out further.
	higherReady := false
	foundHigher := 0
	for i := logNsPerPx - 1; i >= 0; i-- {
		textures := texturesForLevel(r.textures, i)
		n := sort.Search(len(textures), func(j int) bool {
			return textures[j].End() >= end
		})
		if n == len(textures) {
			continue
		}
		f := textures[n]
		if f.computed.done == nil {
			continue
		}
		if f.start <= start {
			out = append(out, f)
			foundHigher++
			if f.ready() {
				// Don't collect more textures that'll never be used.
				higherReady = true
				renderTrace("    found ready higher resolution texture, not collecting any more")
				break
			}
		}
	}

	renderTrace("    found %d higher resolution textures", foundHigher)

	if higherReady {
		// A higher resolution texture is already ready. There is no point in looking for lower resolution ones, or
		// computing an exact match.
		renderTrace("    found higher resolution texture that is already ready to use, returning early")
		return out
	}

	// Find a less zoomed in texture that covers our time range. We can upsample it on the GPU to get a blurry preview
	// of the final texture.
	foundLower := 0
	for i := logNsPerPx + 1; i < 64+logOffset; i++ {
		textures := texturesForLevel(r.textures, i)
		n := sort.Search(len(textures), func(j int) bool {
			return textures[j].End() >= end
		})
		if n == len(textures) {
			continue
		}
		f := textures[n]
		if f.computed.done == nil {
			continue
		}
		if f.start <= start {
			out = append(out, f)
			foundLower++
			if f.ready() {
				// Don't collect more textures that'll never be used.
				renderTrace("    found ready lower resolution texture, not collecting any more")
				break
			}
		}
	}

	renderTrace("    found %d lower resolution textures", foundLower)

	{
		// OPT(dh): avoiding this allocation would be nice
		placeholderTex := instantUniform(&texture{
			track:     track,
			spans:     spans,
			start:     start,
			nsPerPx:   float64(end-start) / texWidth,
			level:     uint8(logNsPerPx),
			ephemeral: true,
		}, placeholderUniform, placeholderOp)
		out = append(out, placeholderTex)
	}

	if foundHigher > 0 || foundExact {
		// We're already waiting on either the exact match, or a higher resolution texture. In neither case do we want
		// to start computing the exact match (again.)
		renderTrace("    found an exact or higher resolution texture that is being computed, not starting another computation")
		return out
	}

	if tex != nil {
		renderTrace("    exact match had its computed data deleted")
	} else {
		tex = &texture{
			track:   track,
			spans:   spans,
			start:   start,
			nsPerPx: nsPerPx,
			level:   uint8(logNsPerPx),
		}
	}
	texsStart, texsEnd := texturesForLevelIndices(r.textures, logNsPerPx)
	n := sort.Search(len(r.textures[texsStart:texsEnd]), func(i int) bool {
		return r.textures[texsStart+i].start >= start
	})
	r.textures = slices.Insert(r.textures, texsStart+n, tex)
	out = slices.Insert(out, 0, tex)

	return out
}

type pixel struct {
	sum       color.LinearSRGB
	sumWeight float64
}

// computeTexture computes a texture for the time range [start, start + texWidth * nsPerPx].
func computeTexture(tex *texture) image.Image {
	debugTexturesComputing.Add(1)
	defer debugTexturesComputing.Add(-1)

	renderTrace("Computing texture at %d ns @ %f ns/px", tex.start, tex.nsPerPx)

	// The texture covers the time range [start, end]
	end := trace.Timestamp(math.Ceil(float64(tex.start) + tex.nsPerPx*texWidth))

	if tex.nsPerPx == 0 {
		panic("got zero nsPerPx")
	}

	first := sort.Search(tex.spans.Len(), func(i int) bool {
		return tex.spans.At(i).End >= tex.start
	})
	last := sort.Search(tex.spans.Len(), func(i int) bool {
		return tex.spans.At(i).Start >= end
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
			// case <-cancelled:
			// 	renderTrace("Cancelled computing texture at %d ns @ %f ns/px", start, nsPerPx)
			// 	return nil
			default:
			}
		}
		span := tex.spans.At(i)

		firstBucket := float64(span.Start-tex.start) / tex.nsPerPx
		lastBucket := float64(span.End-tex.start) / tex.nsPerPx

		if firstBucket >= texWidth {
			break
		}
		if lastBucket < 0 {
			continue
		}

		firstBucket = max(firstBucket, 0)
		lastBucket = min(lastBucket, texWidth)

		colorIdx := tex.track.SpanColor(tex.spans.At(i), tex.track.parent.cv.trace)
		c := mappedColors[colorIdx]

		if int(firstBucket) == int(lastBucket) {
			// falls into a single bucket
			w := float64(span.Duration()) / tex.nsPerPx
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
	// case <-cancelled:
	// 	renderTrace("Cancelled computing texture at %d ns @ %f ns/px", start, nsPerPx)
	// 	return nil
	default:
	}

	img := image.NewRGBA(image.Rect(0, 0, texWidth, 1))
	// Cache the conversion of the final pixel colors to sRGB.
	srgbCache := map[color.LinearSRGB]stdcolor.RGBA{}
	firstSet := false
	allSame := true
	var firstColor stdcolor.RGBA
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
		if firstSet {
			if firstColor != srgb {
				allSame = false
			}
		} else {
			firstSet = true
			firstColor = srgb
		}
		s[0] = srgb.R
		s[1] = srgb.G
		s[2] = srgb.B
		s[3] = srgb.A
	}

	// Turn single-color textures into uniforms. That's a compression ratio of texWidth.
	if allSame {
		return image.NewUniform(firstColor)
	}

	return img
}

// Render returns a series of textures that when placed next to each other will display spans for the time range [start,
// end]. Each texture contains instructions for applying scaling and offsetting, and the series of textures may have
// gaps, expecting the background to already look as desired.
func (r *Renderer) Render(
	win *theme.Window,
	track *Track,
	spans Items[ptrace.Span],
	nsPerPx float64,
	start trace.Timestamp,
	end trace.Timestamp,
	out []TextureStack,
) []TextureStack {
	if c, ok := spans.Container(); ok {
		renderTrace("Requesting texture for [%d ns, %d ns] @ %f ns/px in a track in timeline %q", start, end, nsPerPx, c.Timeline.shortName)
	} else {
		renderTrace("Requesting texture for [%d ns, %d ns] @ %f ns/px", start, end, nsPerPx)
	}

	if nsPerPx == 0 {
		panic("got zero nsPerPx")
	}

	if spans.Len() == 0 {
		return out
	}

	if spans.At(0).State == statePlaceholder {
		// Don't go through the normal pipeline for making textures if the spans are placeholders (which happens when we
		// are dealing with compressed tracks.). Caching them would be wrong, as cached textures don't get invalidated
		// when spans change, and computing them is trivial.
		renderTrace("  returning uniform texture for stack placeholder")

		newStart := max(start, spans.At(0).Start)
		newEnd := min(end, spans.At(0).End)
		if newStart >= end || newEnd <= start {
			return nil
		}
		// OPT(dh): avoiding this allocation would be nice
		tex := instantUniform(&texture{
			track:     track,
			spans:     spans,
			start:     newStart,
			nsPerPx:   float64(newEnd-newStart) / texWidth,
			level:     uint8(math.Log2(nsPerPx)),
			ephemeral: true,
		}, stackPlaceholderUniform, stackPlaceholderOp)
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
		renderTrace("  not returning textures for empty time interval [%d, %d]", start, end)
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
		texs = r.planTextures(win, track, spans, start, nsPerPx, texs[:0])

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

type TextureManager struct {
	Stats RendererStatistics

	rgbas         *mysync.Mutex[*container.RBTree[comparableTimeDuration, *texture]]
	realizedRGBAs *mysync.Mutex[Set[*texture]]
}

func (tm *TextureManager) Render(
	win *theme.Window,
	track *Track,
	spans Items[ptrace.Span],
	nsPerPx float64,
	start trace.Timestamp,
	end trace.Timestamp,
	out []TextureStack,
) []TextureStack {
	return track.rnd.Render(win, track, spans, nsPerPx, start, end, out)
}

func (tm *TextureManager) Image(win *theme.Window, tr *Trace, tex *texture) (image.Image, paint.ImageOp, bool) {
	tex.lastUse = win.Frame
	if tex.realized.done == nil {
		tm.realize(tex, tr)
	}

	if CanRecv(tex.realized.done) {
		return tex.realized.image, tex.realized.op, true
	} else {
		return nil, paint.ImageOp{}, false
	}
}

func (tm *TextureManager) uncompute(texs []*texture) {
	var sz int
	for _, tex := range texs {
		sz += len(tex.computed.compressed)
		tex.computed = textureComputed{}
	}

	tm.Stats.CompressedNum.Add(uint64(-len(texs)))
	tm.Stats.CompressedSize.Add(uint64(-sz))
}

func (tm *TextureManager) unrealize(texs []*texture) {
	s, unlock := tm.realizedRGBAs.Lock()
	defer unlock.Unlock()
	for _, tex := range texs {
		tex.realized = textureRealized{}
		s.Delete(tex)
	}
	tm.Stats.RealizedRGBAs.Add(uint64(-len(texs)))
}

func (tm *TextureManager) realize(tex *texture, tr *Trace) {
	tex.realized.done = make(chan struct{})
	renderTrace("realizing texture at %d ns @ %f ns/px for track in %q", tex.start, tex.nsPerPx, tex.track.parent.shortName)

	if CanRecv(tex.computed.done) && tex.computed.uniform != nil {
		// Some textures are immediately turned into uniforms during the planning phase and don't need concurrent
		// computation or realization.
		renderTrace("  realizing uniform early")
		tex.realized.image = tex.computed.uniform
		tex.realized.op = paint.NewImageOp(tex.computed.uniform)
		if !tex.ephemeral {
			tm.Stats.RealizedUniforms.Add(1)
		}
		close(tex.realized.done)
		return
	}

	if tex.computed.done == nil {
		tm.compute(tex, tr)
	}

	go func() {
		defer close(tex.realized.done)
		<-tex.computed.done
		if tex.computed.uniform != nil {
			tex.realized.image = tex.computed.uniform
			tex.realized.op = paint.NewImageOp(tex.computed.uniform)
			if !tex.ephemeral {
				tm.Stats.RealizedUniforms.Add(1)
			}
		} else {
			pix := make([]byte, texWidth*4)
			r := flate.NewReader(bytes.NewReader(tex.computed.compressed))
			io.ReadFull(r, pix)
			if err := r.Close(); err != nil {
				panic(fmt.Sprintf("error decompressing texture: %s", err))
			}
			tex.realized.image = &image.RGBA{
				Pix:    pix,
				Stride: 4,
				Rect:   image.Rect(0, 0, texWidth, 1),
			}
			tex.realized.op = paint.NewImageOp(tex.realized.image)
			tm.Stats.RealizedRGBAs.Add(1)
			s, unlock := tm.realizedRGBAs.Lock()
			defer unlock.Unlock()
			s.Add(tex)
			if !tex.tracked {
				rgbas, unlock := tm.rgbas.Lock()
				defer unlock.Unlock()
				rgbas.Insert(comparableTimeDuration(tex.computed.computedIn), tex)
				tex.tracked = true
			}
		}
	}()
}

func (tm *TextureManager) compute(tex *texture, tr *Trace) {
	tex.computed.done = make(chan struct{})

	go func() {
		defer close(tex.computed.done)
		if debugSlowRenderer {
			// Simulate a slow renderer.
			time.Sleep(time.Duration(rand.Intn(1000)+3000) * time.Millisecond)
		}

		t := time.Now()
		img := computeTexture(tex)
		// XXX only set this once because of the bst
		tex.computed.computedIn = time.Since(t)
		switch img := img.(type) {
		case *image.Uniform:
			tex.computed.uniform = img
		case *image.RGBA:
			buf := bytes.NewBuffer(nil)
			// We benchmarked planar RLE vs packed DEFLATE. Compression, RLE was slightly faster at worse compression.
			// Decompression, DEFLATE was 3x faster. However, compress/flate allocates significant amounts of memory in
			// flate.NewReader, so this might be worth reconsidering.
			w := flateWriterPool.Get().(*flate.Writer)
			if w == nil {
				w, _ = flate.NewWriter(buf, flate.BestSpeed)
			} else {
				w.Reset(buf)
			}
			defer flateWriterPool.Put(w)
			w.Write(img.Pix)
			w.Close()
			tex.computed.compressed = buf.Bytes()
			tm.Stats.CompressedNum.Add(1)
			tm.Stats.CompressedSize.Add(uint64(len(buf.Bytes())))
		// case nil:
		// 	// Cancelled
		// 	return textureCompressed{}
		default:
			panic(fmt.Sprintf("unexpected type %T", img))
		}
	}()
}

func instantUniform(tex *texture, uniform *image.Uniform, op paint.ImageOp) *texture {
	tex.computed = textureComputed{
		uniform: uniform,
		done:    closedDoneChannel,
	}
	tex.realized = textureRealized{
		image: uniform,
		op:    op,
		done:  closedDoneChannel,
	}
	return tex
}
