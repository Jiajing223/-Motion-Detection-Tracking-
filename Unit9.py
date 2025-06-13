from tkinter import messagebox, filedialog
import cv2 as cv
import numpy as np
import tkinter as tk
import threading

# ================= INTERFACE UI FUNCTIONS ==========================

def center_window(window, width=400, height=150):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    window.geometry(f"{width}x{height}+{x}+{y}")


def get_video_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path


def get_image_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path


def open_video_prompt(option_name, processing_func):
    video_path = get_video_path()
    if not video_path:
        messagebox.showerror("Error", "No video file selected")
        return

    window_name = f"{option_name} - Press Q to quit"
    

    def process_video():
        processing_func(video_path, window_name)
        cv.destroyAllWindows()

    threading.Thread(target=process_video).start()


def open_image_prompt(option_name, processing_func=None):
    image_path = get_image_path()
    if not image_path:
        messagebox.showerror("Error", "No image file selected")
        return

    window_name = f"{option_name} - Press Q to quit"

    if processing_func:
        processing_func(image_path, window_name)
    else:
        img = cv.imread(image_path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image at:\n{image_path}")
            return

        window_name = f"{option_name} - Press any key to close"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 800, 600)
        cv.imshow(window_name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()


# ============== PROCESSING FUNCTIONS ==============

def background_subtractor_simple(video_path, window_name):
    
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video:\n{video_path}")
        return

    ret, prev_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Could not read first frame")
        return

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(prev_gray, gray)
        _, thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

        cv.imshow(window_name, thresh)
        prev_gray = gray

        key = cv.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyWindow(window_name)


def background_subtractor_knn(video_path, window_name):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    cap = cv.VideoCapture(video_path)
    
    subtractor = cv.createBackgroundSubtractorKNN()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask = subtractor.apply(frame)
        cv.putText(fg_mask, "Algorithm: KNN", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

        cv.imshow(window_name, fg_mask)

        key = cv.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv.destroyWindow(window_name)

def background_subtractor_mog2(video_path, window_name):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    cap = cv.VideoCapture(video_path)
    
    subtractor = cv.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask = subtractor.apply(frame)
        cv.putText(fg_mask, "Algorithm: MOG2", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

        cv.imshow(window_name, fg_mask)

        key = cv.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv.destroyWindow(window_name)


def meanshift_tracking(video_path, window_name):
    cap = cv.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        return

    r = cv.selectROI("Select Object for MeanShift", frame, fromCenter=False, showCrosshair=True)
    cv.destroyWindow("Select Object for MeanShift")
    x, y, w, h = r
    track_window = (x, y, w, h)

    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        result = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv.putText(result, "Algorithm: MeanShift", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow(window_name, result)

        key = cv.waitKey(30) & 0xFF
        if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyWindow(window_name)


def camshift_tracking(video_path, window_name):
    cap = cv.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        return

    r = cv.selectROI("Select Object for CAMShift", frame, fromCenter=False, showCrosshair=True)
    cv.destroyWindow("Select Object for CAMShift")
    x, y, w, h = r
    track_window = (x, y, w, h)

    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply CAMShift to get the rotated rectangle
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        
        # Draw rotated rectangle
        pts = cv.boxPoints(ret)
        pts = np.int32(pts)
        result = cv.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        cv.putText(result, "Algorithm: CAMShift", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow(window_name, result)

        key = cv.waitKey(30) & 0xFF
        if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyWindow(window_name)


def harris_corner_detection(image_path, window_name):
    img = cv.imread(image_path)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  

    window_name = "Harris Corner Detection"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyWindow(window_name)

def good_features_corner_detection(image_path, window_name):
    img = cv.imread(image_path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.10, minDistance=100)


    corners = np.int8(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)  

    window_name = "Good Features to Track"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


def lucas_kanade_flow(video_path, window_name):
    cap = cv.VideoCapture(video_path)
    ret, old_frame = cap.read()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv.add(frame, mask)
        cv.putText(img, "Algorithm: Lucas-Kanade Optical Flow", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow(window_name, img)

        key = cv.waitKey(30) & 0xFF
        if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv.destroyWindow(window_name)


def farneback_flow(video_path, window_name):
    cap = cv.VideoCapture(video_path)
    ret, frame1 = cap.read()

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prvs, next, None,
                                           pyr_scale=0.5, levels=3, winsize=15,
                                           iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame2)
        hsv[..., 1] = 255

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.putText(bgr, "Algorithm: Farneback Optical Flow", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow(window_name, bgr)

        key = cv.waitKey(30) & 0xFF
        if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

        prvs = next

    cap.release()
    cv.destroyWindow(window_name)


# ================== MAIN APPLICATION ==================

root = tk.Tk()
root.title("OpenCV Image/Video Processor")
root.geometry("800x600")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Configure grid for 3 columns and 3 rows
for i in range(4):
    main_frame.columnconfigure(i, weight=1)
for i in range(4):
    main_frame.rowconfigure(i, weight=1)

# Updated functions list with separate buttons for MeanShift and CAMShift
functions = [
    ("1. Background Subtraction", lambda: open_video_prompt("Background Subtraction", background_subtractor_simple)),
    ("2. BackgroundSubtractor(KNN)", lambda: open_video_prompt("BackgroundSubtractor (KNN)", background_subtractor_knn)),
    ("2. BackgroundSubtractor(MOG2)", lambda: open_video_prompt("BackgroundSubtractor (MOG2)", background_subtractor_mog2)),
    ("3. MeanShift Tracking", lambda: open_video_prompt("MeanShift Tracking", meanshift_tracking)),
    ("3. CAMShift Tracking", lambda: open_video_prompt("CAMShift Tracking", camshift_tracking)),
    ("4. Harris Corner Detector", lambda: open_image_prompt("Corner Detection", harris_corner_detection)),
    ("4. Good Features to Track", lambda: open_image_prompt("Corner Detection", good_features_corner_detection)),
    ("5. Lucas-Kanade Flow", lambda: open_video_prompt("Lucas-Kanade", lucas_kanade_flow)),
    ("6. Farneback Flow", lambda: open_video_prompt("Farneback", farneback_flow)),
]

# Create buttons in a 3x3 grid
for i, (title, command) in enumerate(functions):
    btn = tk.Button(main_frame, text=title, command=command, height=2)
    row = i // 4
    col = i % 4
    btn.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

root.mainloop()