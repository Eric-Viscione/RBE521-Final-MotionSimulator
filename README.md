# RBE521-Final-MotionSimulator
Gathering Data:
To gather accelerometer data to be used,  the data must be in a csv format with columns of 'time', 'seconds_elapsed', 'z', 'y', 'x'
 - z y and x refer to the axis of acceleration 
 - The data for this project was gathered using the "Sensor Logger" Application https://www.tszheichoi.com/sensorlogger
 - Data was gathered on various rides and the demonstration uses data from Rock 'n' Roller Coaster Starring Aerosmith in Disney Worlds Hollywood Studios recorded on September 27th 2024 in Row 5 Seat 1
1. Install all needed requirments by running 'pip install -r requirments.txt'  
2. To generate a new trajectory from accelerations, point the accelerations.py script to the csv in the "accelerometer_file" variable and then run the script
    - The script will show you the extracted acceleration data, then the proposed magnitudes of pitch, roll and z offset. 
3. The pose_trajectory will then be generated 
4. Use a video that matches the measurment and the recording start time 
5. In the visualize_stewart script, using the "video_file" variable point it to the video file in any standard opencv2 readable format
6. Run the visualize_stewart.py script and enjoy 
