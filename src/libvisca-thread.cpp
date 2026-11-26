#include <obs-module.h>
#include <util/platform.h>
#include <util/threading.h>
#include <graphics/vec2.h>
#include <graphics/graphics.h>
#include <climits>
#include <cstdlib>
#include "plugin-macros.generated.h"
#include "libvisca-thread.hpp"
#include "libvisca.h"
#ifdef _WIN32
#include <winsock.h>
#endif
#ifndef _WIN32
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif

#define debug(...) // blog(LOG_INFO, __VA_ARGS__)

#define TH_FAIL 4

libvisca_thread::libvisca_thread()
{
	debug("libvisca_thread::libvisca_thread");
	iface = NULL;
	camera = NULL;
	data = NULL;
	pan_rsvd = 0;
	tilt_rsvd = 0;
	zoom_rsvd = 0;
	zoom_got = 0;
	pthread_mutex_init(&mutex, 0);

	add_ref(); // release inside thread_main
	pthread_t thread;
	pthread_create(&thread, NULL, libvisca_thread::thread_main, (void *)this);
	pthread_detach(thread);
}

libvisca_thread::~libvisca_thread()
{
	if (iface) {
		VISCA_close(iface);
		bfree(iface);
	}
	if (camera)
		bfree(camera);
	if (data)
		obs_data_release(data);
	pthread_mutex_destroy(&mutex);
}

// libvisca waits for the camera's reply with a blocking recv() and no timeout. A single reply lost on
// the network would then block this thread forever and no further PTZ command would be sent.
// Give the socket a receive/send timeout so a lost reply is counted as a failure, which the loop
// below already answers with a reconnect. Also send each small command at once (no Nagle delay).
// The TCP context of libvisca starts with the socket handle.
static void tune_tcp_socket(struct _VISCA_interface *iface)
{
	if (!iface || !iface->ctx)
		return;
#ifdef _WIN32
	SOCKET fd = *(SOCKET *)iface->ctx;
	DWORD tv = 700;
	BOOL one = TRUE;
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
	setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof(tv));
	setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
#else
	int fd = *(int *)iface->ctx;
	struct timeval tv;
	tv.tv_sec = 0;
	tv.tv_usec = 700000;
	int one = 1;
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
	setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
	setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
}

void libvisca_thread::thread_connect()
{
	pthread_mutex_lock(&mutex);

	const char *address = obs_data_get_string(data, "address");
	int port = (int)obs_data_get_int(data, "port");
	auto *iface_new = (struct _VISCA_interface *)bzalloc(sizeof(struct _VISCA_interface));
	debug("libvisca_thread::thread_connect connecting to address=%s port=%d...", address, port);
	if (VISCA_open_tcp(iface_new, address, port) != VISCA_SUCCESS) {
		blog(LOG_ERROR, "failed to connect %s:%d", address, port);
		bfree(iface_new);
		iface_new = NULL;
	}
	debug("libvisca_thread::thread_connect connected.");
	pthread_mutex_unlock(&mutex);
	if (!iface_new) {
		os_sleep_ms(500); // do not retry in a tight loop while the camera is unreachable
		return;
	}
	tune_tcp_socket(iface_new);

	if (iface) {
		VISCA_close(iface);
		bfree(iface);
	}
	iface = iface_new;
	if (!camera)
		camera = (VISCACamera_t *)bzalloc(sizeof(VISCACamera_t));
	data_changed = false;

	debug("libvisca_thread::thread_connect sending VISCA_clear...");
	camera->address = 1;
	VISCA_clear(iface, camera);
	debug("libvisca_thread::thread_connect exiting successfully");
}

void *libvisca_thread::thread_main(void *data)
{
	auto *visca = (libvisca_thread *)data;

	// add_ref() was called just before creating this thread.

	visca->thread_loop();

	visca->release();

	return NULL;
}

static inline bool send_pantilt(struct _VISCA_interface *iface, struct _VISCA_camera *camera, int pan, int tilt,
				int retry = 0)
{
	debug("send_pantilt moving pan=%d tilt=%d", pan, tilt);
	int pan_a = abs(pan);
	int tilt_a = abs(tilt);
	if (pan_a > 127)
		pan_a = 127;
	if (tilt_a > 127)
		tilt_a = 127;

	uint32_t res = VISCA_SUCCESS;
	if (tilt < 0 && pan < 0) // 1=up, 1=left
		res = VISCA_set_pantilt_upleft(iface, camera, pan_a, tilt_a);
	else if (tilt < 0 && pan == 0) // 1=up
		res = VISCA_set_pantilt_up(iface, camera, pan_a, tilt_a);
	else if (tilt < 0 && pan > 0) // 1=up, 2=right
		res = VISCA_set_pantilt_upright(iface, camera, pan_a, tilt_a);
	else if (tilt == 0 && pan < 0) // 1=left
		res = VISCA_set_pantilt_left(iface, camera, pan_a, tilt_a);
	else if (tilt == 0 && pan == 0)
		res = VISCA_set_pantilt_stop(iface, camera, pan_a, tilt_a);
	else if (tilt == 0 && pan > 0) // 2=right
		res = VISCA_set_pantilt_right(iface, camera, pan_a, tilt_a);
	else if (tilt > 0 && pan < 0) // 2=down, 1=left
		res = VISCA_set_pantilt_downleft(iface, camera, pan_a, tilt_a);
	else if (tilt > 0 && pan == 0) // 2=down
		res = VISCA_set_pantilt_down(iface, camera, pan_a, tilt_a);
	else if (tilt > 0 && pan > 0)
		res = VISCA_set_pantilt_downright(iface, camera, pan_a, tilt_a);

	if (res != VISCA_SUCCESS)
		return false;

	if (iface->type == VISCA_RESPONSE_ERROR && retry < 3) {
		blog(LOG_INFO, "send_pantilt(%d, %d): retrying (%d)", pan, tilt, retry);
		return send_pantilt(iface, camera, pan, tilt, retry + 1);
	}

	return true;
}

static inline bool send_zoom(struct _VISCA_interface *iface, struct _VISCA_camera *camera, int zoom, int retry = 0)
{
	debug("send_zoom moving zoom=%d", zoom);
	uint32_t res;
	int zoom_a = std::abs(zoom);
	if (zoom_a > 7)
		zoom_a = 7;
	// zoom>0 : wide
	if (zoom > 0)
		res = VISCA_set_zoom_wide_speed(iface, camera, zoom_a);
	else if (zoom < 0)
		res = VISCA_set_zoom_tele_speed(iface, camera, zoom_a);
	else
		res = VISCA_set_zoom_stop(iface, camera);

	if (res != VISCA_SUCCESS)
		return false;

	if (iface->type == VISCA_RESPONSE_ERROR && retry < 3) {
		blog(LOG_INFO, "send_zoom(%d): retrying (%d)", zoom, retry);
		return send_zoom(iface, camera, zoom, retry + 1);
	}

	return true;
}

void libvisca_thread::thread_loop()
{
	int pan_prev = INT_MIN, tilt_prev = INT_MIN, zoom_prev = INT_MIN;
	int n_fail = 0;

	while (get_ref() > 1) {
		if (data_changed || n_fail > TH_FAIL) {
			thread_connect();
			n_fail = 0;
			pan_prev = INT_MIN;
			tilt_prev = INT_MIN;
			zoom_prev = INT_MIN;
		}
		if (!iface) {
			os_sleep_ms(50);
			continue;
		}
		int pan = os_atomic_load_long(&pan_rsvd);
		int tilt = os_atomic_load_long(&tilt_rsvd);
		int zoom = os_atomic_load_long(&zoom_rsvd);
		bool ptz_changed = false;
		if (pan != pan_prev || tilt != tilt_prev) {
			if (send_pantilt(iface, camera, pan, tilt)) {
				pan_prev = pan;
				tilt_prev = tilt;
				ptz_changed = true;
				n_fail = 0;
			} else {
				n_fail++;
			}
		}
		if (zoom != zoom_prev) {
			if (send_zoom(iface, camera, zoom)) {
				zoom_prev = zoom;
				ptz_changed = true;
				n_fail = 0;
			} else {
				n_fail++;
			}
		}

		if (os_atomic_set_bool(&preset_changed, false)) {
			os_sleep_ms(48);
			debug("libvisca_thread::thread_loop recall preset=%d", (int)preset_rsvd);
			VISCA_memory_recall(iface, camera, preset_rsvd);
			os_sleep_ms(48);
		}

		if (zoom != 0) {
			uint16_t zoom_cur = 0;
			if (VISCA_get_zoom_value(iface, camera, &zoom_cur) == VISCA_SUCCESS) {
				if (zoom_cur != zoom_got) {
					debug("libvisca_thread::thread_loop got zoom=%d", (int)zoom_cur);
				}
				os_atomic_set_long(&zoom_got, (long)zoom_cur);
			}
		}

		if (!ptz_changed)
			os_sleep_ms(50);
	}
}

void libvisca_thread::set_config(struct obs_data *data_)
{
	pthread_mutex_lock(&mutex);

	obs_data_addref(data_);
	if (data) {
		obs_data_release(data);
		const char *address_old = obs_data_get_string(data, "address");
		int port_old = (int)obs_data_get_int(data, "port");
		const char *address_new = obs_data_get_string(data_, "address");
		int port_new = (int)obs_data_get_int(data_, "port");
		if (strcmp(address_old, address_new))
			data_changed = true;
		if (port_old != port_new)
			data_changed = true;
	} else
		data_changed = true;
	data = data_;

	pthread_mutex_unlock(&mutex);
}

float libvisca_thread::raw2zoomfactor(int zoom)
{
	// TODO: configurable
	return expf((float)zoom * (logf(20.0f) / 16384.f));
}

bool libvisca_thread::ptz_type_modified(obs_properties_t *pp, obs_data_t *settings)
{
	(void)settings;
	if (obs_properties_get(pp, "ptz.visca-over-tcp.address"))
		return false;

	obs_properties_add_text(pp, "ptz.visca-over-tcp.address", obs_module_text("IP address"), OBS_TEXT_DEFAULT);
	obs_properties_add_int(pp, "ptz.visca-over-tcp.port", obs_module_text("Port"), 1, 65535, 1);
	return true;
}
