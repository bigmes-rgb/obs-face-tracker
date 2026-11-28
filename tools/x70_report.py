#!/usr/bin/env python3
"""Tuning report for the X70 head's person tracking.

    python3 x70_report.py <OBS log file> [head "Recent events" text file]

Reads the "X70-STATS" lines the Face Tracker plugin writes once a minute, and its presenter events, and says in plain
words how the tracking went and which setting to change. Nothing is changed by this script.
"""
import re, sys

def parse(path):
    stats, events = [], []
    for line in open(path, errors='replace'):
        m = re.search(r'X70-STATS (.*)', line)
        if m:
            d = {}
            for k, v in re.findall(r'(\w+)=([-\d.]+)', m.group(1)):
                d[k] = float(v)
            stats.append(d)
        elif re.search(r'presenter|walked out of the picture|camera response|Encoding overloaded|person detector|body pose', line):
            events.append(line.strip())
    return stats, events

def report(stats, events, head_lines):
    out, advice = [], []
    if not stats:
        return ['No X70-STATS lines in this log. They are written once a minute while tracking runs with "Whole '
                'person" and zone framing on (plugin build 23 or newer).'], []
    mins = sum(s['secs'] for s in stats) / 60
    seen = sum(s['seen'] * s['secs'] for s in stats) / sum(s['secs'] for s in stats)
    lost = int(sum(s['lost'] for s in stats)); edge = int(sum(s['lost_edge'] for s in stats)); fast = int(sum(s['lost_fast'] for s in stats))
    chased = int(sum(s['chased'] for s in stats)); gone = max(s['gone_max'] for s in stats)
    off = sum(s['off_avg'] for s in stats) / len(stats); offmax = max(s['off_max'] for s in stats)
    vmax = max(s['v_max'] for s in stats); cmd = max(s['cmd_max'] for s in stats)
    pose = sum(s['pose'] for s in stats) / len(stats); moving = sum(s['moving'] for s in stats) / len(stats)
    gp = stats[-1]['gain_pan']; last = stats[-1]
    out.append('Tracked for %.0f minutes. The presenter was in view %.1f%% of that time.' % (mins, seen))
    out.append('Lost %d time(s): %d near the side of the picture, %d while walking. Longest absence %.1f s. Followed out of the picture %d time(s).' % (lost, edge, fast, gone, chased))
    out.append('Framing: on average %.1f%% of the picture width from the framing point, worst %.0f%%. Fastest walk %.0f%% of the width per second.' % (off, offmax, vmax))
    out.append('The camera was turning %.0f%% of the time. Head found from the body pose %.0f%% of the time. People in view at most %d.' % (moving, pose, max(s['people_max'] for s in stats)))
    out.append('Measured head response: %.2f (1.00 = as designed).' % gp)
    per_h = lost / max(mins / 60, 1e-6)
    if lost and edge >= 0.6 * lost and fast >= 0.5 * lost:
        if gp < 0.5:
            advice.append('Most losses are a walking presenter leaving at the side, and the head answers at only %.0f%% of the designed speed. The head itself is the limit: set the travel limits and gear ratios on the head page (without them it is held to a slow safe speed) and raise "Top speed" there.' % (gp * 100))
        elif last['strength'] < 1.6:
            advice.append('Most losses are a walking presenter leaving at the side. Set Movement > "Speed" to Firm.')
        elif last['zone'] > 12:
            advice.append('Most losses are a walking presenter leaving at the side, already on Firm. Set "Sensitivity" to High, so the camera starts sooner.')
        else:
            advice.append('Losses at the side continue on the firmest settings. The shot is too tight for how fast this presenter walks: zoom out a little.')
        if last['pursuit'] < 4 and chased < edge:
            advice.append('Also set "If the person walks out of the picture" to keep turning for 4 seconds.')
    elif lost and edge < 0.4 * lost:
        advice.append('Most losses happened in the middle of the picture, not at the edge: the person was hidden or not recognised (light, someone crossing, the music stand). Check that the orange search area covers the whole stage and try Detection quality = Best.')
    if lost == 0 or per_h < 2:
        if off < 4 and moving > 40:
            advice.append('Nothing was lost and the camera was moving %.0f%% of the time. For a calmer picture set "Sensitivity" one step lower (Medium -> Low).' % moving)
        elif not advice:
            advice.append('No change needed: losses are rare and framing is close.')
    if offmax > 35 and lost == 0:
        advice.append('The presenter got as far as %.0f%% from the framing point without being lost. "Speed" = Firm would keep them closer to the middle.' % offmax)
    if pose < 50:
        advice.append('The body pose was found less than half the time, so the head is mostly guessed from the top of the body box. The presenter is probably small in the picture or half hidden.')
    if any('Encoding overloaded' in e for e in events):
        advice.append('OBS reported "Encoding overloaded": the PC is too busy. Set Detection quality to Fast, or "Look for the person every" to 0.2 s.')
    ms = [int(m.group(1)) for e in events for m in [re.search(r'detection takes about (\d+) ms', e)] if m]
    if ms and sorted(ms)[len(ms) // 2] > 80:
        advice.append('Finding the person takes about %d ms a look, which is slow. Set Detection quality to Fast.' % sorted(ms)[len(ms) // 2])
    if head_lines:
        off_n = sum('OFFLINE' in l for l in head_lines); rst = sum('Started' in l or 'start-up' in l.lower() for l in head_lines)
        wifi = sum('Wi-Fi' in l and ('lost' in l.lower() or 'reconnect' in l.lower()) for l in head_lines)
        out.append('Head: %d restart(s), %d Wi-Fi drop(s), joystick box offline %d time(s).' % (rst, wifi, off_n))
        if rst > 1: advice.append('The head restarted during the session. Send the "Recent events" text: the start-up reason tells whether it was power or a crash.')
        if wifi or off_n: advice.append('Wi-Fi dropped during the session. Move the head/joystick closer to the access point or use the stronger of the two saved networks.')
    return out, advice

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    stats, events = parse(sys.argv[1])
    head = [l.strip() for l in open(sys.argv[2], errors='replace')] if len(sys.argv) > 2 else []
    out, advice = report(stats, events, head)
    print('\n'.join(out)); print('\nWhat to change:' if advice else '')
    for a in advice: print(' - ' + a)
