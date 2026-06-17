"""
Fall Detection GUI System
=========================
Professional GUI built with tkinter + OpenCV.
Uses fall_detection_model.pt (local weights).

Requirements:
    pip install ultralytics opencv-python pillow
"""

import cv2
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import deque
import datetime

# =====================================
# Configuration
# =====================================
MODEL_LOCAL_PATH = "fall_detection_model.pt"
CONF_THRESHOLD   = 0.25
FALL_ALERT_DELAY = 4       # seconds
ALERT_COOLDOWN   = 10      # seconds


# ============================================================
#  COLOUR PALETTE  (dark tactical theme)
# ============================================================
BG        = "#0d1117"
BG2       = "#161b22"
BG3       = "#21262d"
BORDER    = "#30363d"
ACCENT    = "#e05252"       # alert red
ACCENT2   = "#58a6ff"       # info blue
GREEN     = "#3fb950"
YELLOW    = "#d29922"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"
FONT_HEAD = ("Courier New", 11, "bold")
FONT_BODY = ("Courier New", 10)
FONT_BIG  = ("Courier New", 26, "bold")
FONT_MED  = ("Courier New", 13, "bold")


# ============================================================
#  HELPER  – rounded pill label
# ============================================================
def pill(parent, text, bg, fg=TEXT, **kw):
    f = tk.Frame(parent, bg=bg, bd=0)
    tk.Label(f, text=text, bg=bg, fg=fg,
             font=FONT_BODY, padx=8, pady=2).pack()
    return f


# ============================================================
#  MAIN APPLICATION
# ============================================================
class FallDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fall Detection System")
        self.root.configure(bg=BG)
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── state ──────────────────────────────────────────
        self.model          = None
        self.cap            = None
        self.video_path     = tk.StringVar(value="No video selected")
        self.running        = False
        self.paused         = False
        self._thread        = None

        self.conf_var       = tk.DoubleVar(value=CONF_THRESHOLD)
        self.delay_var      = tk.IntVar(value=FALL_ALERT_DELAY)

        # stats
        self.total_frames   = 0
        self.fall_frames    = 0
        self.total_alerts   = 0
        self.fps_history    = deque(maxlen=30)
        self.session_start  = None

        # fall timer
        self.fall_start_t   = None
        self.alert_fired    = False
        self.last_alert_t   = 0.0
        self._alert_active  = False

        # event log
        self.log_entries    = []

        self._build_ui()
        self._load_model_async()

    # =========================================================
    #  UI BUILD
    # =========================================================
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────
        topbar = tk.Frame(self.root, bg=BG2, height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡  FALL DETECTION SYSTEM",
                 font=("Courier New", 15, "bold"),
                 bg=BG2, fg=ACCENT).pack(side="left", padx=20, pady=10)

        self.model_badge = tk.Label(topbar, text="● MODEL LOADING…",
                                    font=FONT_BODY, bg=BG2, fg=YELLOW)
        self.model_badge.pack(side="right", padx=20)

        self.clock_lbl = tk.Label(topbar, text="", font=FONT_BODY,
                                  bg=BG2, fg=TEXT_DIM)
        self.clock_lbl.pack(side="right", padx=10)
        self._tick_clock()

        # ── main paned layout ─────────────────────────────────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=(6, 10))

        # left column (video + controls)
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        # right column (stats + log)
        right = tk.Frame(main, bg=BG, width=290)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self._build_video_panel(left)
        self._build_controls(left)
        self._build_right_panel(right)

    # ── video canvas ─────────────────────────────────────────
    def _build_video_panel(self, parent):
        frame = tk.Frame(parent, bg=BORDER, bd=1)
        frame.pack(fill="both", expand=True)

        inner = tk.Frame(frame, bg=BG3)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.canvas = tk.Canvas(inner, bg="#000000",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # placeholder text
        self._placeholder()

        # alert banner (hidden by default)
        self.alert_banner = tk.Frame(inner, bg=ACCENT, height=36)
        self.alert_label  = tk.Label(self.alert_banner,
                                     text="🚨  FALL DETECTED — EMERGENCY ALERT TRIGGERED",
                                     font=("Courier New", 11, "bold"),
                                     bg=ACCENT, fg="white")
        self.alert_label.pack(pady=6)

    def _placeholder(self):
        self.canvas.delete("placeholder")
        self.canvas.create_text(
            self.canvas.winfo_reqwidth() // 2 or 500,
            self.canvas.winfo_reqheight() // 2 or 280,
            text="[ SELECT A VIDEO TO BEGIN ]",
            fill=TEXT_DIM,
            font=("Courier New", 14),
            tags="placeholder"
        )

    # ── controls bar ─────────────────────────────────────────
    def _build_controls(self, parent):
        bar = tk.Frame(parent, bg=BG2, height=110)
        bar.pack(fill="x", pady=(6, 0))
        bar.pack_propagate(False)

        # row 1 – file picker + buttons
        row1 = tk.Frame(bar, bg=BG2)
        row1.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(row1, text="VIDEO:", font=FONT_HEAD,
                 bg=BG2, fg=TEXT_DIM).pack(side="left")

        self.path_lbl = tk.Label(row1, textvariable=self.video_path,
                                 font=FONT_BODY, bg=BG2, fg=ACCENT2,
                                 width=46, anchor="w")
        self.path_lbl.pack(side="left", padx=(6, 12))

        self.btn_browse = self._btn(row1, "📂 Browse", self._browse,
                                    ACCENT2, "#0d1117")
        self.btn_browse.pack(side="left", padx=3)

        self.btn_start = self._btn(row1, "▶  Start", self._start,
                                   GREEN, "#0d1117")
        self.btn_start.pack(side="left", padx=3)

        self.btn_pause = self._btn(row1, "⏸  Pause", self._toggle_pause,
                                   YELLOW, "#0d1117", state="disabled")
        self.btn_pause.pack(side="left", padx=3)

        self.btn_stop = self._btn(row1, "⏹  Stop", self._stop,
                                  ACCENT, "white", state="disabled")
        self.btn_stop.pack(side="left", padx=3)

        # row 2 – sliders
        row2 = tk.Frame(bar, bg=BG2)
        row2.pack(fill="x", padx=12)

        self._slider_group(row2, "Confidence:", self.conf_var,
                           0.05, 0.95, "{:.0%}")
        tk.Frame(row2, bg=BORDER, width=1).pack(side="left",
                                                fill="y", padx=12)
        self._slider_group(row2, "Alert Delay:", self.delay_var,
                           1, 10, "{:.0f}s")

    def _slider_group(self, parent, label, var, from_, to, fmt):
        tk.Label(parent, text=label, font=FONT_BODY,
                 bg=BG2, fg=TEXT_DIM).pack(side="left")
        val_lbl = tk.Label(parent, text=fmt.format(var.get()),
                           font=FONT_HEAD, bg=BG2, fg=TEXT, width=5)
        val_lbl.pack(side="left", padx=(4, 6))

        def on_change(_):
            val_lbl.config(text=fmt.format(var.get()))

        s = ttk.Scale(parent, variable=var,
                      from_=from_, to=to,
                      orient="horizontal", length=130,
                      command=on_change)
        s.pack(side="left", padx=(0, 8))

    # ── right panel ──────────────────────────────────────────
    def _build_right_panel(self, parent):
        # ── live stats ───────────────────────────────────────
        stats_frame = tk.LabelFrame(parent,
                                    text="  LIVE STATS  ",
                                    font=FONT_HEAD,
                                    bg=BG2, fg=TEXT_DIM,
                                    bd=1, relief="flat",
                                    labelanchor="n")
        stats_frame.pack(fill="x", pady=(0, 8))

        self.stat_vars = {}
        rows = [
            ("STATUS",   "IDLE",   TEXT_DIM),
            ("FPS",      "—",      ACCENT2),
            ("FRAMES",   "0",      TEXT),
            ("FALLS",    "0",      ACCENT),
            ("ALERTS",   "0",      ACCENT),
            ("DURATION", "00:00",  TEXT),
        ]
        for key, default, color in rows:
            r = tk.Frame(stats_frame, bg=BG2)
            r.pack(fill="x", padx=10, pady=3)
            tk.Label(r, text=key, font=FONT_BODY,
                     bg=BG2, fg=TEXT_DIM, width=10,
                     anchor="w").pack(side="left")
            lbl = tk.Label(r, text=default, font=FONT_HEAD,
                           bg=BG2, fg=color, anchor="e")
            lbl.pack(side="right")
            self.stat_vars[key] = (lbl, color)

        # fall indicator
        self.fall_indicator = tk.Label(parent,
                                       text="NO FALL DETECTED",
                                       font=("Courier New", 10, "bold"),
                                       bg=GREEN, fg="white",
                                       pady=6)
        self.fall_indicator.pack(fill="x", pady=(0, 8))

        # countdown bar
        cnt_frame = tk.Frame(parent, bg=BG2)
        cnt_frame.pack(fill="x", pady=(0, 8))
        tk.Label(cnt_frame, text="ALERT COUNTDOWN",
                 font=FONT_BODY, bg=BG2, fg=TEXT_DIM).pack(anchor="w", padx=8)
        self.countdown_lbl = tk.Label(cnt_frame, text="—",
                                      font=("Courier New", 28, "bold"),
                                      bg=BG2, fg=ACCENT2)
        self.countdown_lbl.pack()
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Red.Horizontal.TProgressbar",
                        troughcolor=BG3, background=ACCENT,
                        thickness=10)
        self.progress = ttk.Progressbar(cnt_frame,
                                        variable=self.progress_var,
                                        maximum=100,
                                        style="Red.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=8, pady=(4, 8))

        # ── event log ────────────────────────────────────────
        log_frame = tk.LabelFrame(parent,
                                  text="  EVENT LOG  ",
                                  font=FONT_HEAD,
                                  bg=BG2, fg=TEXT_DIM,
                                  bd=1, relief="flat",
                                  labelanchor="n")
        log_frame.pack(fill="both", expand=True)

        self.log_box = tk.Text(log_frame, bg=BG3, fg=TEXT,
                               font=("Courier New", 9),
                               state="disabled", wrap="word",
                               relief="flat", bd=0,
                               insertbackground=TEXT)
        sb = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

        self.log_box.tag_config("fall",  foreground=ACCENT)
        self.log_box.tag_config("ok",    foreground=GREEN)
        self.log_box.tag_config("info",  foreground=ACCENT2)
        self.log_box.tag_config("time",  foreground=TEXT_DIM)

        # clear log btn
        tk.Button(parent, text="Clear Log",
                  font=FONT_BODY, bg=BG3, fg=TEXT_DIM,
                  relief="flat", bd=0,
                  command=self._clear_log).pack(pady=(4, 0))

    # =========================================================
    #  WIDGET HELPERS
    # =========================================================
    def _btn(self, parent, text, cmd, bg, fg, state="normal"):
        b = tk.Button(parent, text=text, command=cmd,
                      font=FONT_BODY, bg=bg, fg=fg,
                      relief="flat", bd=0,
                      padx=10, pady=5,
                      activebackground=bg,
                      activeforeground=fg,
                      state=state,
                      cursor="hand2")
        return b

    def _set_btn_state(self, btn, state):
        btn.config(state=state)

    # =========================================================
    #  CLOCK
    # =========================================================
    def _tick_clock(self):
        now = datetime.datetime.now().strftime("%H:%M:%S  %d %b %Y")
        self.clock_lbl.config(text=now)
        self.root.after(1000, self._tick_clock)

    # =========================================================
    #  MODEL LOADING
    # =========================================================
    def _load_model_async(self):
        def _load():
            if not os.path.exists(MODEL_LOCAL_PATH):
                self.root.after(0, lambda: self.model_badge.config(
                    text="● MODEL NOT FOUND", fg=ACCENT))
                self._log("Model file not found: " + MODEL_LOCAL_PATH, "fall")
                return
            try:
                m = YOLO(MODEL_LOCAL_PATH)
                self.model = m
                classes = list(m.names.values())
                self.root.after(0, lambda: self.model_badge.config(
                    text=f"● MODEL READY  [{', '.join(classes)}]", fg=GREEN))
                self._log(f"Model loaded — classes: {classes}", "ok")
            except Exception as e:
                self.root.after(0, lambda: self.model_badge.config(
                    text="● MODEL ERROR", fg=ACCENT))
                self._log(f"Model error: {e}", "fall")

        threading.Thread(target=_load, daemon=True).start()

    # =========================================================
    #  FILE BROWSE
    # =========================================================
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")]
        )
        if path:
            self.video_path.set(os.path.basename(path))
            self._full_video_path = path
            self._log(f"Video selected: {os.path.basename(path)}", "info")

    # =========================================================
    #  START / PAUSE / STOP
    # =========================================================
    def _start(self):
        if not self.model:
            messagebox.showerror("Error", "Model is not loaded yet.")
            return
        path = getattr(self, "_full_video_path", None)
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        if self.running:
            return

        self.running        = True
        self.paused         = False
        self.total_frames   = 0
        self.fall_frames    = 0
        self.total_alerts   = 0
        self.fall_start_t   = None
        self.alert_fired    = False
        self.session_start  = time.time()

        self._set_btn_state(self.btn_start, "disabled")
        self._set_btn_state(self.btn_pause, "normal")
        self._set_btn_state(self.btn_stop,  "normal")
        self._set_btn_state(self.btn_browse,"disabled")

        self._update_stat("STATUS", "RUNNING", GREEN)
        self._log("Session started", "ok")

        self._thread = threading.Thread(target=self._video_loop, daemon=True)
        self._thread.start()

    def _toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶  Resume")
            self._update_stat("STATUS", "PAUSED", YELLOW)
            self._log("Paused", "info")
        else:
            self.btn_pause.config(text="⏸  Pause")
            self._update_stat("STATUS", "RUNNING", GREEN)
            self._log("Resumed", "info")

    def _stop(self):
        self.running = False
        self.paused  = False
        self._set_btn_state(self.btn_start, "normal")
        self._set_btn_state(self.btn_pause, "disabled")
        self._set_btn_state(self.btn_stop,  "disabled")
        self._set_btn_state(self.btn_browse,"normal")
        self.btn_pause.config(text="⏸  Pause")
        self._update_stat("STATUS", "STOPPED", ACCENT)
        self._hide_alert_banner()
        self.fall_indicator.config(text="NO FALL DETECTED", bg=GREEN)
        self.countdown_lbl.config(text="—", fg=ACCENT2)
        self.progress_var.set(0)
        self._log("Session stopped", "info")

    # =========================================================
    #  VIDEO PROCESSING LOOP (runs in thread)
    # =========================================================
    def _video_loop(self):
        cap = cv2.VideoCapture(self._full_video_path)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror(
                "Error", "Cannot open video file."))
            self._stop()
            return

        fps_raw  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # output writer
        folder      = os.path.dirname(self._full_video_path)
        output_path = os.path.join(folder, "fall_detection_output.mp4")
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_raw,
            (width, height)
        )

        conf  = self.conf_var.get()
        prev  = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            self.total_frames += 1
            now  = time.time()
            dt   = now - prev
            prev = now
            fps  = 1.0 / max(dt, 1e-6)
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            # ── inference ───────────────────────────────────
            results = model.predict(
                frame,
                conf=self.conf_var.get(),
                verbose=False
            ) if (model := self.model) else []

            annotated        = frame.copy()
            fall_in_frame    = False

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id     = int(box.cls[0])
                    conf_val   = float(box.conf[0])
                    class_name = self.model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    is_fall = class_name.lower() == "fallen"
                    color   = (0, 0, 255) if is_fall else (0, 220, 80)

                    if is_fall:
                        fall_in_frame = True

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated,
                                f"{class_name} {conf_val:.2f}",
                                (x1, max(y1 - 10, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

            if fall_in_frame:
                self.fall_frames += 1

            # ── fall timer ──────────────────────────────────
            delay = float(self.delay_var.get())

            if fall_in_frame:
                if self.fall_start_t is None:
                    self.fall_start_t = now
                    self.alert_fired  = False
                    self.root.after(0, lambda: self._log(
                        "Fall detected — timer started", "fall"))

                elapsed   = now - self.fall_start_t
                remaining = max(0.0, delay - elapsed)
                pct       = min(100.0, (elapsed / delay) * 100)

                self.root.after(0, lambda r=remaining, p=pct:
                    self._update_countdown(r, p))
                self.root.after(0, lambda:
                    self.fall_indicator.config(
                        text="⚠  FALL IN PROGRESS", bg=ACCENT))
                self.root.after(0, self._show_alert_banner)

                if elapsed >= delay and not self.alert_fired:
                    if now - self.last_alert_t >= ALERT_COOLDOWN:
                        self.alert_fired  = True
                        self.last_alert_t = now
                        self.total_alerts += 1
                        self.root.after(0, self._popup_alert)
                        self.root.after(0, lambda:
                            self._log("🚨 EMERGENCY ALERT TRIGGERED", "fall"))

            else:
                if self.fall_start_t is not None:
                    self.root.after(0, lambda:
                        self._log("Person recovered", "ok"))
                self.fall_start_t = None
                self.alert_fired  = False
                self.root.after(0, lambda:
                    self.fall_indicator.config(
                        text="✔  NO FALL DETECTED", bg=GREEN))
                self.root.after(0, self._hide_alert_banner)
                self.root.after(0, lambda:
                    self._update_countdown(None, 0))

            # ── draw FPS overlay on frame ───────────────────
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (200, 200, 200), 2)

            # ── write output ────────────────────────────────
            writer.write(annotated)

            # ── update canvas ───────────────────────────────
            self.root.after(0, lambda f=annotated: self._show_frame(f))

            # ── update stats ────────────────────────────────
            elapsed_session = time.time() - self.session_start
            m, s = divmod(int(elapsed_session), 60)
            self.root.after(0, lambda a=avg_fps, m=m, s=s:
                self._refresh_stats(a, m, s))

        cap.release()
        writer.release()
        self.root.after(0, self._stop)
        self.root.after(0, lambda: self._log(
            f"Output saved: {output_path}", "ok"))

    # =========================================================
    #  CANVAS DISPLAY
    # =========================================================
    def _show_frame(self, frame_bgr):
        if not self.running:
            return
        try:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 2 or ch < 2:
                return
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            fh, fw    = frame_rgb.shape[:2]
            scale     = min(cw / fw, ch / fh)
            nw, nh    = int(fw * scale), int(fh * scale)
            resized   = cv2.resize(frame_rgb, (nw, nh))
            img       = ImageTk.PhotoImage(Image.fromarray(resized))
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2,
                                     anchor="center", image=img)
            self.canvas._img = img   # prevent GC
        except Exception:
            pass

    # =========================================================
    #  STATS HELPERS
    # =========================================================
    def _update_stat(self, key, value, color=None):
        lbl, default_color = self.stat_vars[key]
        lbl.config(text=str(value),
                   fg=color if color else default_color)

    def _refresh_stats(self, fps, minutes, secs):
        self._update_stat("FPS",      f"{fps:.1f}",      ACCENT2)
        self._update_stat("FRAMES",   str(self.total_frames), TEXT)
        self._update_stat("FALLS",    str(self.fall_frames),
                          ACCENT if self.fall_frames else TEXT)
        self._update_stat("ALERTS",   str(self.total_alerts),
                          ACCENT if self.total_alerts else TEXT)
        self._update_stat("DURATION", f"{minutes:02d}:{secs:02d}", TEXT)

    def _update_countdown(self, remaining, pct):
        if remaining is None:
            self.countdown_lbl.config(text="—", fg=ACCENT2)
            self.progress_var.set(0)
        else:
            self.countdown_lbl.config(
                text=f"{remaining:.1f}s",
                fg=ACCENT if remaining < 1.5 else YELLOW)
            self.progress_var.set(pct)

    # =========================================================
    #  ALERT BANNER (in-window)
    # =========================================================
    def _show_alert_banner(self):
        if not self.alert_banner.winfo_ismapped():
            self.alert_banner.pack(fill="x", before=self.canvas)

    def _hide_alert_banner(self):
        if self.alert_banner.winfo_ismapped():
            self.alert_banner.pack_forget()

    # =========================================================
    #  EMERGENCY POPUP
    # =========================================================
    def _popup_alert(self):
        if self._alert_active:
            return
        self._alert_active = True

        win = tk.Toplevel(self.root)
        win.title("EMERGENCY ALERT")
        win.configure(bg="#8B0000")
        win.geometry("500x300")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"+{sw//2-250}+{sh//2-150}")

        tk.Label(win, text="🚨  FALL DETECTED  🚨",
                 font=("Courier New", 20, "bold"),
                 bg="#8B0000", fg="white").pack(pady=(28, 6))

        tk.Label(win,
                 text="A person has fallen!\nContact emergency services immediately.",
                 font=("Courier New", 12),
                 bg="#8B0000", fg="white",
                 justify="center").pack(pady=8)

        tk.Label(win, text="📞  1122  /  115  /  1021",
                 font=("Courier New", 15, "bold"),
                 bg="#8B0000", fg="yellow").pack(pady=6)

        def dismiss():
            self._alert_active = False
            win.destroy()

        tk.Button(win, text="✔  DISMISS",
                  font=("Courier New", 12, "bold"),
                  bg="white", fg="#8B0000",
                  relief="flat", padx=20, pady=8,
                  command=dismiss).pack(pady=16)

        win.protocol("WM_DELETE_WINDOW", dismiss)

    # =========================================================
    #  EVENT LOG
    # =========================================================
    def _log(self, message: str, tag: str = "info"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{ts}] ", "time")
        self.log_box.insert("end", message + "\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    # =========================================================
    #  CLOSE
    # =========================================================
    def _on_close(self):
        self.running = False
        self.root.after(300, self.root.destroy)


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()

    # ── dark title-bar on Windows ──────────────────────────
    try:
        from ctypes import windll, byref, sizeof, c_int
        HWND = windll.user32.GetParent(root.winfo_id())
        windll.dwmapi.DwmSetWindowAttribute(
            HWND, 20, byref(c_int(2)), sizeof(c_int))
    except Exception:
        pass

    app = FallDetectionApp(root)
    root.mainloop()
