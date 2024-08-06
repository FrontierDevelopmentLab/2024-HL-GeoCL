# Setting Up InfluxDB on Google Cloud Platform Using Google Click to Deploy

This guide provides instructions to set up InfluxDB on Google Cloud Platform (GCP) using Google Click to Deploy. Additionally, it includes steps to SSH into the VM and install Python 3.

## Step 1: Launch InfluxDB Using Google Click to Deploy

1. **Open Google Cloud Console**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).

2. **Search for InfluxDB**:
   - In the search bar at the top of the console, type `InfluxDB Google Click to Deploy`.

3. **Launch InfluxDB**:
   - You will see a screen similar to the screenshot below. Click on the **Launch** button.
<img width="440" alt="Screenshot 2024-08-06 at 3 18 42 PM" src="https://github.com/user-attachments/assets/2ad3fafb-5e07-4fac-81b3-6f5cbc142f68">

4. **Configure InfluxDB Deployment**:
   - Follow the on-screen instructions to configure your InfluxDB deployment.
   - Choose your preferred region, zone, machine type, and other settings as needed.
   - Click **Deploy** to start the deployment process.

[![Watch the video](https://img.youtube.com/vi/_5tFXJQIzi4/0.jpg)]([https://www.youtube.com/watch?v=_5tFXJQIzi4](https://youtu.be/bNL6DvtmD40))

## Step 2: Access the InfluxDB VM

1. **View Deployments**:
   - Once the deployment is complete, click on **View Deployments**.
   - Find your InfluxDB deployment in the list and click on it.

2. **SSH into the VM**:
   - Click on the **SSH** button next to your InfluxDB VM instance to open an SSH terminal.

## Step 3: Update System Packages

1. **Update and Upgrade Packages**:
   ```sh
   sudo apt-get update
   sudo apt-get upgrade -y

## Step 4: Install Python 3 on Your VM

1. **SSH into your VM**:
   - In the Google Cloud Console, navigate to the VM instances page.
   - Find your InfluxDB instance and click on the SSH button to open a terminal session to your VM.

2. **Update package lists and install Python 3**:
   ```sh
   sudo apt-get update
   sudo apt-get install python3
3. **Verify Python installation**:
   ```sh
   python3 --version

## Existing InfluxDB Login Information

- **URL**: [https://34.48.13.92:8086](https://34.48.13.92:8086)
- **Login**: `admin`
- **Password**: `nJVFJby7tE2fd18p`

You have successfully set up InfluxDB on Google Cloud Platform using Google Click to Deploy and installed Python 3 via SSH. For more detailed configurations, you can refer to the [InfluxDB documentation](https://docs.influxdata.com/influxdb/v2.0/).

