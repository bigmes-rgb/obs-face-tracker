#pragma once
// The person detector itself, free of OBS so that it can be tested on its own.
#include <algorithm>
#include <cmath>
#include <vector>
#include <dlib/pixel.h>
#include <dlib/matrix.h>
#include <net.h>
#include "presenter-tracker.hpp"

#define PD_INPUT_SIZE 416 // the network's input, pixels square
#define PD_CONF_MIN 0.15f // boxes below this confidence are dropped here; the presenter lock applies its own limits
#define PD_NMS_IOU 0.45f

static inline float pd_box_iou(const pt_det &a, const pt_det &b)
{
	const float ix = std::min(a.x1, b.x1) - std::max(a.x0, b.x0);
	const float iy = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);
	if (ix <= 0.0f || iy <= 0.0f)
		return 0.0f;
	const float inter = ix * iy;
	return inter / ((a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter);
}

// Colours of the clothing: a coarse colour histogram of the middle of the upper body. Used by the presenter lock
// to tell people apart. (x0..y1 are in the coordinates of img.)
static inline void pd_clothing_signature(const dlib::matrix<dlib::rgb_pixel> &img, pt_det &d, float x0, float y0,
					 float x1, float y1)
{
	const float w = x1 - x0, h = y1 - y0;
	const int ax0 = std::max(0, (int)(x0 + 0.25f * w)), ax1 = std::min((int)img.nc(), (int)(x1 - 0.25f * w));
	const int ay0 = std::max(0, (int)(y0 + 0.20f * h)), ay1 = std::min((int)img.nr(), (int)(y0 + 0.60f * h));
	d.has_sig = false;
	for (int i = 0; i < PT_SIG; i++)
		d.sig[i] = 0.0f;
	if (ax1 - ax0 < 2 || ay1 - ay0 < 2 || (ax1 - ax0) * (ay1 - ay0) < 20)
		return;
	int n = 0;
	for (int y = ay0; y < ay1; y++) {
		for (int x = ax0; x < ax1; x++) {
			const dlib::rgb_pixel &px = img(y, x);
			d.sig[(px.red >> 6) * 16 + (px.green >> 6) * 4 + (px.blue >> 6)] += 1.0f;
			n++;
		}
	}
	for (int i = 0; i < PT_SIG; i++)
		d.sig[i] /= (float)n;
	d.has_sig = true;
}

// Finds the people inside the search area (cx0..cy1, in pixels of img). The boxes are returned in the coordinates
// of the full-size picture (img is 1/scale of it).
static inline bool person_detect_core(ncnn::Net &net, const dlib::matrix<dlib::rgb_pixel> &img, int cx0, int cy0,
				      int cx1, int cy1, float scale, std::vector<pt_det> &result)
{
	const int cw = cx1 - cx0, ch = cy1 - cy0;
	// Fit the search area into the network's square input, keeping its shape (grey bars fill the rest).
	const float r = std::min((float)PD_INPUT_SIZE / (float)cw, (float)PD_INPUT_SIZE / (float)ch);
	const int tw = std::max(1, (int)(cw * r)), th = std::max(1, (int)(ch * r));
	const unsigned char *pixels = (const unsigned char *)&img(cy0, cx0);
	ncnn::Mat resized =
		ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_RGB2BGR, cw, ch, (int)img.nc() * 3, tw, th);
	ncnn::Mat padded;
	ncnn::copy_make_border(resized, padded, 0, PD_INPUT_SIZE - th, 0, PD_INPUT_SIZE - tw, ncnn::BORDER_CONSTANT,
			       114.0f);

	// The network's first step ("focus"): each 2x2 block of pixels becomes four values side by side.
	const int half = PD_INPUT_SIZE / 2;
	ncnn::Mat in(half, half, 12);
	static const int dy[4] = {0, 1, 0, 1}, dx[4] = {0, 0, 1, 1};
	for (int g = 0; g < 4; g++) {
		for (int c = 0; c < 3; c++) {
			const ncnn::Mat src = padded.channel(c);
			float *dst = in.channel(g * 3 + c);
			for (int y = 0; y < half; y++) {
				const float *row = src.row(y * 2 + dy[g]);
				for (int x = 0; x < half; x++)
					dst[y * half + x] = row[x * 2 + dx[g]];
			}
		}
	}

	ncnn::Mat out;
	{
		ncnn::Extractor ex = net.create_extractor();
		ex.input("in0", in);
		if (ex.extract("out0", out) != 0 || out.w < 6)
			return false;
	}

	// Each row of the result is one candidate box: x, y, w, h, objectness, then one score per kind of object.
	// Kind 0 is "person".
	std::vector<pt_det> cand;
	int row = 0;
	static const int strides[3] = {8, 16, 32};
	for (int si = 0; si < 3 && row < out.h; si++) {
		const int s = strides[si], n = PD_INPUT_SIZE / s;
		for (int gy = 0; gy < n && row < out.h; gy++) {
			for (int gx = 0; gx < n && row < out.h; gx++, row++) {
				const float *o = out.row(row);
				const float score = o[4] * o[5];
				if (score < PD_CONF_MIN)
					continue;
				const float bx = (o[0] + gx) * s, by = (o[1] + gy) * s;
				const float bw = expf(o[2]) * s, bh = expf(o[3]) * s;
				pt_det d;
				d.x0 = (bx - bw * 0.5f) / r;
				d.y0 = (by - bh * 0.5f) / r;
				d.x1 = (bx + bw * 0.5f) / r;
				d.y1 = (by + bh * 0.5f) / r;
				d.score = score;
				d.has_sig = false;
				cand.push_back(d);
			}
		}
	}
	std::sort(cand.begin(), cand.end(), [](const pt_det &a, const pt_det &b) { return a.score > b.score; });
	std::vector<pt_det> keep;
	for (size_t i = 0; i < cand.size() && keep.size() < 32; i++) {
		bool dup = false;
		for (size_t j = 0; j < keep.size() && !dup; j++)
			dup = pd_box_iou(cand[i], keep[j]) > PD_NMS_IOU;
		if (!dup)
			keep.push_back(cand[i]);
	}

	// back to the coordinates of the whole picture, with the clothing colours of each person
	for (size_t i = 0; i < keep.size(); i++) {
		pt_det &d = keep[i];
		d.x0 = std::max(0.0f, d.x0);
		d.y0 = std::max(0.0f, d.y0);
		d.x1 = std::min((float)cw, d.x1);
		d.y1 = std::min((float)ch, d.y1);
		pd_clothing_signature(img, d, d.x0 + cx0, d.y0 + cy0, d.x1 + cx0, d.y1 + cy0);
		d.x0 = (d.x0 + cx0) * scale;
		d.y0 = (d.y0 + cy0) * scale;
		d.x1 = (d.x1 + cx0) * scale;
		d.y1 = (d.y1 + cy0) * scale;
	}
	result.swap(keep);
	return true;
}
