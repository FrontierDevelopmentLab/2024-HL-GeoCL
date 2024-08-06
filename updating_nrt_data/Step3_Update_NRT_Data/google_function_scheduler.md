# Automating Data Collection with Google Cloud Functions and Scheduler

## Overview

This guide explains how to set up Google Cloud Functions and Google Cloud Scheduler to automate the data collection for ACE, DSCOVR, and Geomagnetic Indices. We will show how these services work together to create scheduled tasks and provide a space for a video demonstration.

## Video Demonstration

Below is a video that shows the connection between Google Cloud Functions and Google Cloud Scheduler, and how they are used to create automated schedules.

[![Google Cloud Functions and Scheduler](https://img.youtube.com/vi/ry-16qxzIZM/0.jpg)](https://youtu.be/ry-16qxzIZM)

## Setting Up Google Cloud Scheduler and Functions

### Step-by-Step Guide

1. **Create Google Cloud Functions**:
    - Write your Python scripts for data collection.
    - Deploy these scripts as Google Cloud Functions.
    - Ensure each function is triggered via HTTP.

2. **Set Up Google Cloud Scheduler**:
    - Go to the Google Cloud Console.
    - Navigate to **Cloud Scheduler**.
    - Create a new job for each data collection task.
    - Set the frequency using cron syntax.
    - Set the target to the HTTP endpoint of the corresponding Google Cloud Function.

### Example Crontab File for Google Cloud Scheduler
#### DISCLAIMER Kp, Hp30, ap30, and Fadj require query strings at the end of their url, for example: https://yourgooglefunction.com?index=kp

```plaintext
# ACE and DSCOVR data collection every 5 minutes
*/5 * * * *

# Kp index data collection every 3 hours
0 */3 * * *

# Hp30 and ap30 data collection every 30 minutes
*/30 * * * *

# Fadj data collection every 24 hours
0 0 * * *
