
from tkinter import messagebox, filedialog
import cv2 as cv
import numpy as np
import tkinter as tk
import threading


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

def create_control_bar(window_name):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)
    
    control_bar = np.zeros((50, 800, 3), dtype=np.uint8)
    cv.rectangle(control_bar, (700, 10), (790, 40), (0, 0, 255), -1)
    cv.putText(control_bar, "STOP", (710, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return control_bar

def open_video_prompt(option_name, processing_func):
    video_path = get_video_path()
    if not video_path:
        messagebox.showerror("Error", "No video file selected")
        return

    window_name = f"{option_name} - Press Q to quit"
    control_bar = create_control_bar(window_name)
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)

    stop_flag = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal stop_flag
        if event == cv.EVENT_LBUTTONDOWN:
            if 700 <= x <= 790 and 10 <= y <= 40:
                stop_flag = True
    
    cv.setMouseCallback(window_name, mouse_callback)
    
    def process_video():
        processing_func(video_path, stop_flag, window_name, control_bar)
        cv.destroyAllWindows()
    
    threading.Thread(target=process_video).start()

def open_image_prompt(option_name, processing_func=None):
    image_path = get_image_path()
    if not image_path:
        messagebox.showerror("Error", "No image file selected")
        return
    
    if processing_func:
        processing_func(image_path)
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

# ============== EMPTY PROCESSING FUNCTIONS ==============

def background_subtractor_simple(video_path, stop_flag, window_name, control_bar):
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
        
        cv.imshow('Simple Background Subtraction', thresh)
        prev_gray = gray
        
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

def background_subtractor_class(video_path, stop_flag, window_name, control_bar):
    cap = cv.VideoCapture(0)
    subtractors = {
        "MOG2": cv.createBackgroundSubtractorMOG2(),
        "KNN": cv.createBackgroundSubtractorKNN(),
        "GMG": cv.createBackgroundSubtractorGMG()
    }
    
    print("Press: 1-MOG2, 2-KNN, 3-GMG, q-Quit")
    current = "MOG2"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        fg_mask = subtractors[current].apply(frame)
        c2.putText(fg_mask, f"Algorithm: {current}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv.imshow('Background Subtractor Class', fg_mask)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current = "MOG2"
        elif key == ord('2'):
            current = "KNN"
        elif key == ord('3'):
            current = "GMG"
            
    cap.release()
    cv.destroyAllWindows()

def meanshift_camshift(video_path, stop_flag, window_name, control_bar):
    """Implement MeanShift and CAMShift tracking algorithms"""
    pass

def corner_detection(image_path):
    """Implement Harris Corner Detector and Good Features to Track"""
    pass

def lucas_kanade_flow(video_path, stop_flag, window_name, control_bar):
    """Implement Lucas-Kanade optical flow"""
    pass

def farneback_flow(video_path, stop_flag, window_name, control_bar):
    """Implement Farneback dense optical flow"""
    pass

# ================== MAIN APPLICATION ==================

root = tk.Tk()
root.title("OpenCV Image/Video Processor")
root.geometry("800x600")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

for i in range(3):
    main_frame.columnconfigure(i, weight=1)
for i in range(2):
    main_frame.rowconfigure(i, weight=1)

functions = [
    ("1. Background Subtraction", lambda: open_video_prompt("Background Subtraction", background_subtractor_simple)),
    ("2. BackgroundSubtractor Class", lambda: open_video_prompt("BackgroundSubtractor", background_subtractor_class)),
    ("3. MeanShift/CAMShift", lambda: open_video_prompt("Tracking", meanshift_camshift)),
    ("4. Corner Detection", lambda: open_image_prompt("Corner Detection", corner_detection)),
    ("5. Lucas-Kanade Flow", lambda: open_video_prompt("Lucas-Kanade", lucas_kanade_flow)),
    ("6. Farneback Flow", lambda: open_video_prompt("Farneback", farneback_flow)),
]

for i, (title, command) in enumerate(functions):
    btn = tk.Button(main_frame, text=title, command=command, height=2)
    btn.grid(row=i//3, column=i%3, padx=10, pady=10, sticky="nsew")

root.mainloop()