#pragma once
// The person detector itself, free of OBS so that it can be tested on its own.
#include <algorithm>
#include <cmath>
#include <vector>
#include <net.h>
#include <dlib/pixel.h>
#include <dlib/matrix.h>
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

// ---- Body pose of one person (MoveNet, run on a crop around that person) ---------------------------------------
#define PD_POSE_SIZE 192

// box: the person, in pixels of img. Fills kp with x, y (pixels of img) and confidence for the 17 body points.
static inline bool pd_pose(ncnn::Net &net, const dlib::matrix<dlib::rgb_pixel> &img, float bx0, float by0, float bx1,
			   float by1, float kp[PT_KP][3])
{
	// a crop with room around the person: the model works best when the person does not touch the edges
	const float bw = bx1 - bx0, bh = by1 - by0;
	const int cx0 = std::max(0, (int)(bx0 - 0.6f * bw)), cx1 = std::min((int)img.nc(), (int)(bx1 + 0.6f * bw));
	const int cy0 = std::max(0, (int)(by0 - 0.1f * bh)), cy1 = std::min((int)img.nr(), (int)(by1 + 0.1f * bh));
	const int cw = cx1 - cx0, ch = cy1 - cy0;
	if (cw < 16 || ch < 32)
		return false;
	const float r = std::min((float)PD_POSE_SIZE / (float)cw, (float)PD_POSE_SIZE / (float)ch);
	const int nw = std::max(1, (int)(cw * r)), nh = std::max(1, (int)(ch * r));
	const int px = (PD_POSE_SIZE - nw) / 2, py = (PD_POSE_SIZE - nh) / 2;
	ncnn::Mat resized = ncnn::Mat::from_pixels_resize((const unsigned char *)&img(cy0, cx0), ncnn::Mat::PIXEL_RGB,
							  cw, ch, (int)img.nc() * 3, nw, nh);
	ncnn::Mat in;
	ncnn::copy_make_border(resized, in, py, PD_POSE_SIZE - nh - py, px, PD_POSE_SIZE - nw - px,
			       ncnn::BORDER_CONSTANT, 0.0f);
	static const float mean[3] = {127.5f, 127.5f, 127.5f}, norm[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
	in.substract_mean_normalize(mean, norm);

	ncnn::Mat heat, center, regress, offset;
	{
		ncnn::Extractor ex = net.create_extractor();
		ex.input("input", in);
		if (ex.extract("heatmap", heat) || ex.extract("center", center) || ex.extract("regress", regress) ||
		    ex.extract("offset", offset))
			return false;
	}
	const int f = PD_POSE_SIZE / 4; // the model answers on a 48 x 48 grid
	if (heat.c != f || heat.h != f || heat.w != PT_KP || regress.w != 2 * PT_KP || offset.w != 2 * PT_KP ||
	    (int)center.total() < f * f)
		return false;

	// 1. where the middle of the person is
	const float *ce = center;
	int best = 0;
	for (int i = 1; i < f * f; i++)
		if (ce[i] > ce[best])
			best = i;
	const int cy = best / f, cx = best % f;
	// 2. from there the model points roughly at each body point; 3. the exact spot is the strongest response near it
	const float *rg = regress.channel(cy).row(cx);
	for (int k = 0; k < PT_KP; k++) {
		const float ry = rg[k] + (float)cy, rx = rg[k + PT_KP] + (float)cx;
		float best_w = -1.0f;
		int ky = 0, kx = 0;
		for (int y = 0; y < f; y++) {
			const ncnn::Mat hrow = heat.channel(y);
			for (int x = 0; x < f; x++) {
				const float dy = (float)y - ry, dx = (float)x - rx;
				const float w = hrow.row(x)[k] / (sqrtf(dy * dy + dx * dx) + 1.8f);
				if (w > best_w) {
					best_w = w;
					ky = y;
					kx = x;
				}
			}
		}
		const float *of = offset.channel(ky).row(kx);
		const float yy = ((float)ky + of[k]) / (float)f * PD_POSE_SIZE;
		const float xx = ((float)kx + of[k + PT_KP]) / (float)f * PD_POSE_SIZE;
		kp[k][0] = (xx - (float)px) / r + (float)cx0;
		kp[k][1] = (yy - (float)py) / r + (float)cy0;
		kp[k][2] = heat.channel(ky).row(kx)[k];
	}
	return true;
}

// ---- How much the camera itself moved between two analysed pictures -------------------------------------------
// When the camera pans or tilts, the whole background slides across the picture. Measuring that slide tells us how
// fast the camera is really turning, with no feedback from the head and no calibration. The picture is cut into
// horizontal bands (for left/right) and vertical strips (for up/down); each one votes for a shift and the middle
// vote wins, so the person walking through some of the bands does not disturb the result.
#define PD_EGO_BANDS 6

struct pd_ego_state
{
	std::vector<float> col[PD_EGO_BANDS], row[PD_EGO_BANDS];
	int tw = 0, th = 0;
	bool have = false;
};

// Best shift of 'cur' against 'prev' (cur[x] = prev[x - shift]), in samples. Returns false when the profile has
// no clear match (a plain wall, for example).
static inline bool pd_profile_shift(const std::vector<float> &prev, const std::vector<float> &cur, int max_shift,
				    float &shift)
{
	const int n = (int)cur.size();
	if (n < 32 || (int)prev.size() != n)
		return false;
	std::vector<float> sad(2 * max_shift + 1);
	int best = 0;
	float sum = 0.0f;
	for (int sft = -max_shift; sft <= max_shift; sft++) {
		const int x0 = std::max(1, 1 + sft), x1 = std::min(n - 1, n - 1 + sft);
		float a = 0.0f;
		for (int x = x0; x < x1; x++)
			a += fabsf((cur[x + 1] - cur[x - 1]) - (prev[x - sft + 1] - prev[x - sft - 1]));
		a /= (float)std::max(1, x1 - x0);
		sad[sft + max_shift] = a;
		sum += a;
		if (a < sad[best])
			best = sft + max_shift;
	}
	const float mean = sum / (float)sad.size();
	if (mean <= 1e-3f || sad[best] > 0.6f * mean)
		return false;
	float sub = 0.0f;
	if (best > 0 && best < (int)sad.size() - 1) {
		const float l = sad[best - 1], c = sad[best], r = sad[best + 1];
		const float den = l - 2.0f * c + r;
		if (den > 1e-6f)
			sub = 0.5f * (l - r) / den;
	}
	shift = (float)(best - max_shift) + sub;
	return true;
}

// img: the picture as fed to the network (3 channels, tw x th valid pixels). Returns the background's movement
// since the previous call, in pixels of img.
static inline bool pd_ego_motion(pd_ego_state &st, const ncnn::Mat &img, int tw, int th, float &dx, float &dy)
{
	std::vector<float> col[PD_EGO_BANDS], row[PD_EGO_BANDS];
	const float *c0 = img.channel(0), *c1 = img.channel(1), *c2 = img.channel(2);
	const int stride = img.w;
	for (int b = 0; b < PD_EGO_BANDS; b++) {
		col[b].assign(tw, 0.0f);
		row[b].assign(th, 0.0f);
	}
	for (int y = 0; y < th; y++) {
		const int by = std::min(PD_EGO_BANDS - 1, y * PD_EGO_BANDS / th);
		for (int x = 0; x < tw; x++) {
			const int i = y * stride + x;
			const float g = c0[i] + c1[i] + c2[i];
			col[by][x] += g;
			row[std::min(PD_EGO_BANDS - 1, x * PD_EGO_BANDS / tw)][y] += g;
		}
	}
	bool ok = false;
	if (st.have && st.tw == tw && st.th == th) {
		float vx[PD_EGO_BANDS], vy[PD_EGO_BANDS];
		int nx = 0, ny = 0;
		for (int b = 0; b < PD_EGO_BANDS; b++) {
			float s;
			if (pd_profile_shift(st.col[b], col[b], tw / 5, s))
				vx[nx++] = s;
			if (pd_profile_shift(st.row[b], row[b], th / 5, s))
				vy[ny++] = s;
		}
		if (nx >= 3 && ny >= 3) {
			std::sort(vx, vx + nx);
			std::sort(vy, vy + ny);
			dx = vx[nx / 2];
			dy = vy[ny / 2];
			ok = true;
		}
	}
	for (int b = 0; b < PD_EGO_BANDS; b++) {
		st.col[b].swap(col[b]);
		st.row[b].swap(row[b]);
	}
	st.tw = tw;
	st.th = th;
	st.have = true;
	return ok;
}

// Finds the people inside the search area (cx0..cy1, in pixels of img). The boxes are returned in the coordinates
// of the full-size picture (img is 1/scale of it).
// If 'ego' is given, 'ego_ok' / 'ego_dx' / 'ego_dy' report how far the background moved since the previous call, in
// pixels of the full-size picture.
static inline bool person_detect_core(ncnn::Net &net, const dlib::matrix<dlib::rgb_pixel> &img, int cx0, int cy0,
				      int cx1, int cy1, float scale, std::vector<pt_det> &result,
				      pd_ego_state *ego = nullptr, bool *ego_ok = nullptr, float *ego_dx = nullptr,
				      float *ego_dy = nullptr)
{
	const int cw = cx1 - cx0, ch = cy1 - cy0;
	// Fit the search area into the network's square input, keeping its shape (grey bars fill the rest).
	const float r = std::min((float)PD_INPUT_SIZE / (float)cw, (float)PD_INPUT_SIZE / (float)ch);
	const int tw = std::max(1, (int)(cw * r)), th = std::max(1, (int)(ch * r));
	const unsigned char *pixels = (const unsigned char *)&img(cy0, cx0);
	ncnn::Mat resized =
		ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_RGB2BGR, cw, ch, (int)img.nc() * 3, tw, th);
	if (ego && ego_ok && ego_dx && ego_dy) {
		float dx = 0.0f, dy = 0.0f;
		*ego_ok = pd_ego_motion(*ego, resized, tw, th, dx, dy);
		*ego_dx = dx / r * scale;
		*ego_dy = dy / r * scale;
	}
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
				d.has_kp = false;
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
