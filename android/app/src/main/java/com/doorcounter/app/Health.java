package com.doorcounter.app;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.SystemClock;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/** A snapshot of how the phone is doing: heat, battery, speed. Sent to the gateway with every batch. */
final class Health {
    private static final File THERMAL_DIR = new File("/sys/class/thermal");
    // Snapdragon 801 on-die sensors; the hottest one is the CPU temperature we report.
    private static final String CPU_SENSOR_PREFIX = "tsens_tz_sensor";
    private static final double MILLI = 1000.0;

    final double batteryTempC;
    final int batteryLevelPct;
    final boolean isCharging;
    /** Hottest on-die sensor, or null when the sensors cannot be read. */
    final Double cpuTempC;
    /** Detector frames per second over the last stats window. */
    final double fps;
    final double meanInferenceS;
    final double uptimeS;

    private Health(double batteryTempC, int batteryLevelPct, boolean isCharging, Double cpuTempC, double fps,
                   double meanInferenceS, double uptimeS) {
        this.batteryTempC = batteryTempC;
        this.batteryLevelPct = batteryLevelPct;
        this.isCharging = isCharging;
        this.cpuTempC = cpuTempC;
        this.fps = fps;
        this.meanInferenceS = meanInferenceS;
        this.uptimeS = uptimeS;
    }

    /**
     * Read battery and temperature sensors now.
     *
     * @param context any context of this app
     * @param fps detector frames per second, measured by the caller
     * @param meanInferenceS mean detector time per frame, measured by the caller
     * @return the snapshot
     */
    static Health sample(Context context, double fps, double meanInferenceS) {
        // Sticky broadcast: registering with a null receiver just returns the last battery state.
        Intent battery = context.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        double tempC = battery == null ? 0.0 : battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0;
        int level = battery == null ? 0 : battery.getIntExtra(BatteryManager.EXTRA_LEVEL, 0);
        int plugged = battery == null ? 0 : battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0);
        return new Health(tempC, level, plugged != 0, readCpuTempC(), fps, meanInferenceS,
                SystemClock.elapsedRealtime() / MILLI);
    }

    /** @return the hottest CPU sensor in degrees Celsius, or null if none can be read */
    private static Double readCpuTempC() {
        File[] zones = THERMAL_DIR.listFiles();
        if (zones == null) {
            return null;
        }
        Double hottest = null;
        for (File zone : zones) {
            try {
                if (!readLine(new File(zone, "type")).startsWith(CPU_SENSOR_PREFIX)) {
                    continue;
                }
                double value = Double.parseDouble(readLine(new File(zone, "temp")));
                double tempC = value > MILLI ? value / MILLI : value;  // some kernels report milli-degrees
                if (hottest == null || tempC > hottest) {
                    hottest = tempC;
                }
            } catch (IOException | NumberFormatException e) {
                // A sensor we may not read; the others still count.
            }
        }
        return hottest;
    }

    /**
     * Read the first line of a small sysfs file.
     *
     * @param file the file
     * @return its first line, trimmed
     * @throws IOException if it cannot be read or is empty
     */
    private static String readLine(File file) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line = reader.readLine();
            if (line == null) {
                throw new IOException("empty file " + file);
            }
            return line.trim();
        }
    }

    /**
     * @return this snapshot as the JSON object the gateway expects
     * @throws JSONException never in practice; org.json declares it
     */
    JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject()
                .put("battery_temp_c", batteryTempC)
                .put("battery_level_pct", batteryLevelPct)
                .put("is_charging", isCharging)
                .put("fps", fps)
                .put("mean_inference_s", meanInferenceS)
                .put("uptime_s", uptimeS);
        if (cpuTempC != null) {
            json.put("cpu_temp_c", cpuTempC.doubleValue());
        }
        return json;
    }
}
