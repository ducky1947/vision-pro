# vision-pro
A robust, AI-based surveillance and maintenance solution designed to track individuals and manage data streams from multiple IP, CCTV, or network cameras in real-time. Each imaging pipeline operates independently using multithreaded architecture to ensure fault isolation—if one source fails or crashes, others continue functioning unaffected.

Core Features
1.Multi-Camera Support
  Supports multiple IP/CCTV/network cameras. Each video stream is handled in a separate thread, ensuring isolated processing pipelines.

2.Resilient Threaded Architecture
  Fault-tolerant design: a crash in one thread does not impact the performance or stability of others.

3.High-Accuracy Intrusion Detection
  Utilizes a HOG + CNN-based model achieving 99.76% accuracy for real-time facial recognition and intruder detection.

4.SMTP Integration for Email Alerts
  On detection of intrusions, the system sends email notifications via SMTP with the date, time, and snapshot of the event.

5.Powerful GUI with PyQt5
  Intuitive graphical interface that includes:
    Live monitoring
    Intruder history & movement tracking with image logs
    Text mining features to search/filter historical records
    Control panel to start/stop system operations
    User registration via real-time image capture or encoding updates

6.Data Export Tools
  Export or download surveillance data based on:
    Specific date
    Day, week, month, or year
    Custom date range

7.Interactive Feedback via Dialogs
  UI incorporates temporary dialog messages to provide feedback on operations and outcomes.


