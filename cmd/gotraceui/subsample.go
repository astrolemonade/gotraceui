package main

// FIXME not too unexpectedly, we have seams when stitching textures, probably because of rounding errors and some
// samples not belonging to either texture.

import (
	"image"
	stdcolor "image/color"
	"math"
	"sort"
	"sync"

	"honnef.co/go/gotraceui/color"
	"honnef.co/go/gotraceui/theme"
	"honnef.co/go/gotraceui/tinylfu"
	"honnef.co/go/gotraceui/trace"
	"honnef.co/go/gotraceui/trace/ptrace"
)

const texWidth = 8192

type Renderer struct {
	oklchToLinear *tinylfu.T[color.Oklch, color.LinearSRGB]
}

type Thingy struct {
	mu         sync.Mutex
	populating bool
	Image      *image.RGBA
}

type DisplayTexture struct {
	Image   *image.RGBA
	XScale  float32
	XOffset float32
}

type Texture struct {
	Start   trace.Timestamp
	NsPerPx float64
	Image   *image.RGBA
}

func (tex Texture) End() trace.Timestamp {
	return tex.Start + trace.Timestamp(float64(tex.Image.Rect.Dx())*tex.NsPerPx)
}

func (r *Renderer) renderTexture(start trace.Timestamp, nsPerPx float64, spans Items[ptrace.Span], tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex) Texture {
	if r.oklchToLinear == nil {
		r.oklchToLinear = tinylfu.New[color.Oklch, color.LinearSRGB](1024, 1024*10)
	}

	end := trace.Timestamp(math.Ceil(float64(start) + nsPerPx*texWidth))
	first := sort.Search(spans.Len(), func(i int) bool {
		return spans.At(i).End >= start
	})
	last := sort.Search(spans.Len(), func(i int) bool {
		return spans.At(i).Start >= end
	})
	if last-first < texWidth {
		// XXX return a proper but empty texture
		return Texture{}
	}

	type pixel struct {
		sum       color.LinearSRGB
		sumWeight float64
	}

	pixels := make([]pixel, texWidth)

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

	if last-first <= 0 {
		img := image.NewRGBA(image.Rect(0, 0, texWidth, 1))
		for x := 0; x < texWidth; x++ {
			i := img.PixOffset(x, 0)
			s := img.Pix[i : i+4 : i+4] // Small cap improves performance, see https://golang.org/issue/27857
			s[0] = 255
			s[1] = 255
			s[2] = 235
			s[3] = 255
		}
		return Texture{
			Image:   img,
			Start:   start,
			NsPerPx: nsPerPx,
		}
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
		c, ok := r.oklchToLinear.Get(colors[colorIdx])
		if !ok {
			c := colors[colorIdx].MapToSRGBGamut()
			r.oklchToLinear.Add(colors[colorIdx], c)
		}

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

		srgb := stdcolor.RGBAModel.Convert(px.sum.SRGB()).(stdcolor.RGBA)
		i := img.PixOffset(x, 0)
		s := img.Pix[i : i+4 : i+4] // Small cap improves performance, see https://golang.org/issue/27857
		s[0] = srgb.R
		s[1] = srgb.G
		s[2] = srgb.B
		s[3] = srgb.A
	}

	return Texture{
		Image:   img,
		Start:   start,
		NsPerPx: nsPerPx,
	}
}

func (r *Renderer) Render(win *theme.Window, spans Items[ptrace.Span], nsPerPx float64, tr *Trace, spanColor func(ptrace.Span, *Trace) colorIndex, start trace.Timestamp, end trace.Timestamp, out []DisplayTexture) []DisplayTexture {
	if spanColor == nil {
		spanColor = defaultSpanColor
	}

	origStart := start
	origNsPerPx := nsPerPx
	nsPerPx = math.Pow(2, math.Floor(math.Log2(nsPerPx)))
	m := texWidth * nsPerPx
	start = trace.Timestamp(m * math.Floor(float64(start)/m))

	step := float64(texWidth) * nsPerPx

	// XXX this addition got to have some rounding error
	for start := start; start < end; start += trace.Timestamp(step) {
		tex := r.renderTexture(start, nsPerPx, spans, tr, spanColor)

		out = append(out, DisplayTexture{
			Image:   tex.Image,
			XScale:  float32(nsPerPx / origNsPerPx),
			XOffset: float32(float64(start-origStart) / origNsPerPx),
		})
	}

	return out
}
