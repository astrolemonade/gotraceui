package theme

import (
	"context"
	"image"
	rtrace "runtime/trace"

	"honnef.co/go/gotraceui/color"
	"honnef.co/go/gotraceui/layout"
	"honnef.co/go/gotraceui/widget"

	"gioui.org/font"
	"gioui.org/op"
	"gioui.org/op/clip"
	"gioui.org/unit"
)

type DialogStyle struct {
	BorderWidth     unit.Dp
	BorderColor     color.Oklch
	Title           string
	TitleSize       unit.Sp
	TitleColor      color.Oklch
	TitleBackground color.Oklch
	TitlePadding    unit.Dp
	Background      color.Oklch
	Padding         unit.Dp
}

func Dialog(th *Theme, title string) DialogStyle {
	return DialogStyle{
		BorderWidth:     th.WindowBorder,
		BorderColor:     th.Palette.Border,
		Title:           title,
		TitleSize:       th.TextSize,
		TitleColor:      th.Palette.Popup.TitleForeground,
		TitleBackground: th.Palette.Popup.TitleBackground,
		TitlePadding:    th.WindowPadding,
		Background:      th.Palette.Popup.Background,
		Padding:         th.WindowPadding * 2,
	}
}

func (ds DialogStyle) Layout(win *Window, gtx layout.Context, w Widget) layout.Dimensions {
	defer rtrace.StartRegion(context.Background(), "theme.DialogStyle.Layout").End()

	titleGtx := gtx
	titleGtx.Constraints.Min.Y = 0
	titleGtx.Constraints.Max.X -= 2 * gtx.Dp(ds.BorderWidth+ds.TitlePadding)
	titleGtx.Constraints.Max.Y -= 2 * gtx.Dp(ds.BorderWidth+ds.TitlePadding)
	titleGtx.Constraints = layout.Normalize(titleGtx.Constraints)

	m := op.Record(titleGtx.Ops)
	labelDims := widget.Label{MaxLines: 1}.Layout(titleGtx, win.Theme.Shaper, font.Font{Weight: font.Bold}, ds.TitleSize, ds.Title, win.ColorMaterial(gtx, ds.TitleColor))
	labelCall := m.Stop()

	wGtx := gtx
	wGtx.Constraints.Min.Y -= labelDims.Size.Y
	wGtx.Constraints.Max.X -= 2 * gtx.Dp(ds.BorderWidth+ds.Padding)
	wGtx.Constraints.Max.Y -= 2 * gtx.Dp(ds.BorderWidth+ds.Padding)
	wGtx.Constraints.Max.Y -= labelDims.Size.Y + 2*gtx.Dp(ds.TitlePadding+ds.BorderWidth)
	wGtx.Constraints = layout.Normalize(wGtx.Constraints)

	var wDims layout.Dimensions
	var wCall op.CallOp
	m = op.Record(wGtx.Ops)
	wDims = w(win, wGtx)
	wCall = m.Stop()

	if labelDims.Size.X > wDims.Size.X {
		wDims.Size.X = labelDims.Size.X
	} else if wDims.Size.X > labelDims.Size.X {
		labelDims.Size.X = wDims.Size.X
	}

	return Bordered{Width: ds.BorderWidth, Color: ds.BorderColor}.Layout(win, gtx, func(win *Window, gtx layout.Context) layout.Dimensions {
		return layout.Rigids(gtx, layout.Vertical,
			func(gtx layout.Context) layout.Dimensions {
				defer clip.Rect{Max: image.Pt(wDims.Size.X+gtx.Dp(ds.Padding)*2, labelDims.Size.Y+gtx.Dp(ds.TitlePadding)*2)}.Push(gtx.Ops).Pop()
				Fill(win, gtx.Ops, ds.TitleBackground)
				return layout.UniformInset(ds.TitlePadding).Layout(gtx, func(gtx layout.Context) layout.Dimensions {
					labelCall.Add(gtx.Ops)
					return labelDims
				})
			},

			func(gtx layout.Context) layout.Dimensions {
				defer clip.Rect{Max: image.Pt(wDims.Size.X+gtx.Dp(ds.Padding)*2, wDims.Size.Y+gtx.Dp(ds.Padding)*2)}.Push(gtx.Ops).Pop()
				Fill(win, gtx.Ops, ds.Background)
				return layout.UniformInset(ds.Padding).Layout(gtx, func(gtx layout.Context) layout.Dimensions {
					wCall.Add(gtx.Ops)
					return wDims
				})
			},
		)
	})
}
