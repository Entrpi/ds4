#!/usr/bin/env python3
"""Drive ds4-agent through a pseudo-terminal (its /save, /switch and /quit
live in the interactive UI only).  Each step is sent once the agent is idle:
the pty has been quiet for `quiet` seconds, the prompt is on screen, and --
when `wait_active` -- some output arrived since the previous step (so a turn
that is still loading the model or generating is never interrupted).

usage: agent_pty_drive.py LOG STEPS.json -- ds4-agent ARGS...
STEPS.json: [{"send": "...", "quiet": 25, "timeout": 1200, "wait_active": true}, ...]"""
import fcntl, json, os, pty, re, select, struct, sys, termios, time

def main():
    log_path, steps_path = sys.argv[1], sys.argv[2]
    cmd = sys.argv[sys.argv.index('--') + 1:]
    steps = json.load(open(steps_path))
    pid, fd = pty.fork()
    if pid == 0:
        os.environ['TERM'] = 'xterm-256color'
        os.execvp(cmd[0], cmd)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', 48, 200, 0, 0))
    log = open(log_path, 'wb')
    state = {'buf': b'', 'eof': False}

    def pump(timeout):
        r, _, _ = select.select([fd], [], [], timeout)
        if fd not in r:
            return b''
        try:
            data = os.read(fd, 65536)
        except OSError:
            data = b''
        if not data:
            state['eof'] = True
            return None
        log.write(data); log.flush()
        state['buf'] = (state['buf'] + data)[-200000:]
        return data

    ansi = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][A-Z0-9]|\r')

    def footer_state():
        """The agent's status footer reads `ctx N/M | <state>`; its last redraw
        is the worker's current state (idle, prefill, generating, ...)."""
        tail = ansi.sub('', state['buf'][-20000:].decode('utf-8', 'replace'))
        m = None
        for m in re.finditer(r'ctx [0-9.]+[kM]?/[0-9.]+[kM]? \| ([a-z]+)', tail):
            pass
        return m.group(1) if m else ''

    def wait_idle(quiet, timeout, need_activity):
        """Idle = the footer says idle, nothing was written for `quiet` seconds,
        and (when need_activity) the worker was seen busy or wrote something
        since the last step, so a turn still loading or generating is never
        cut short."""
        t0 = time.time(); last = time.time(); active = not need_activity
        while time.time() - t0 < timeout:
            d = pump(0.5)
            if d is None:
                return False
            if d:
                last = time.time(); active = True
            st = footer_state()
            if st and st != 'idle':
                active = True
            if active and st == 'idle' and time.time() - last >= quiet and 'ds4-agent> ' in \
                    ansi.sub('', state['buf'][-3000:].decode('utf-8', 'replace')):
                return True
        return False

    def wait_expect(pattern, timeout):
        """Block until `pattern` (regex) shows up in the output; a `(y/n)`
        question from the agent is answered no (nothing here needs saving
        twice).  Returns True when seen."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            d = pump(0.5)
            if d is None:
                return False
            tail = ansi.sub('', state['buf'][-8000:].decode('utf-8', 'replace'))
            if re.search(pattern, tail):
                return True
            if tail.rstrip().endswith('(y/n)'):
                os.write(fd, b'n\r')
                print(f"[drive {time.strftime('%H:%M:%S')}] answered n to a (y/n) question", flush=True)
                time.sleep(1.0)
        return False

    for i, st in enumerate(steps):
        need_activity = st.get('wait_active', True)
        if st.get('expect'):
            seen = wait_expect(st['expect'], st.get('timeout', 1200))
            print(f"[drive {time.strftime('%H:%M:%S')}] step {i} expect={st['expect']!r} seen={seen}", flush=True)
            if seen:
                need_activity = False   # the expected output already proved the step ran
        ok = wait_idle(st.get('quiet', 25), st.get('timeout', 1200), need_activity)
        print(f"[drive {time.strftime('%H:%M:%S')}] step {i} idle={ok} footer={footer_state()!r} send={st['send'][:70]!r}", flush=True)
        if state['eof']:
            print('[drive] agent exited early', flush=True)
            break
        os.write(fd, (st['send'] + '\r').encode())   # linenoise raw mode: Enter is CR
        time.sleep(1.5)
    t0 = time.time()
    while not state['eof'] and time.time() - t0 < 180:
        if pump(0.5):
            tail = ansi.sub('', state['buf'][-400:].decode('utf-8', 'replace'))
            if tail.rstrip().endswith('(y/n)'):
                os.write(fd, b'n\r')
                print(f"[drive {time.strftime('%H:%M:%S')}] answered n to a (y/n) question at exit", flush=True)
    if not state['eof']:
        print('[drive] agent did not exit; killing it', flush=True)
        os.kill(pid, 9)
    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        status = 0
    print(f"[drive {time.strftime('%H:%M:%S')}] exit status {status}", flush=True)

if __name__ == '__main__':
    main()
