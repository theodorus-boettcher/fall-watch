package com.doorcounter.app;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

/**
 * Changes the settings from the laptop, then restarts the counter service so they take effect.
 *
 * <p>Guarded by {@code android.permission.DUMP} like {@link ProvisionReceiver}: only the adb shell
 * can reach it. Example: {@code adb shell am broadcast -n com.doorcounter.app/.ConfigReceiver
 * --ei rotation 180 --ez preview true}. Extras left out keep their current value. Extras:
 * {@code rotation} (int), {@code gateway} (string), {@code preview} (boolean), {@code threads} (int),
 * {@code motion_gate} (boolean), {@code model} (string, e.g. {@code models/sort-v1}).
 */
public final class ConfigReceiver extends BroadcastReceiver {
    private static final String TAG = "DoorConfig";

    @Override
    public void onReceive(Context context, Intent intent) {
        Settings current = Settings.load(context);
        try {
            Settings.save(context,
                    intent.getIntExtra("rotation", current.rotationDeg),
                    intent.hasExtra("gateway") ? intent.getStringExtra("gateway") : current.gatewayUrl,
                    intent.getBooleanExtra("preview", current.isPreviewEnabled),
                    intent.getIntExtra("threads", current.threadCount),
                    intent.getBooleanExtra("motion_gate", current.isMotionGateEnabled),
                    intent.hasExtra("model") ? intent.getStringExtra("model") : current.modelName);
        } catch (IllegalArgumentException e) {
            report("error: " + e.getMessage());
            return;
        }
        Intent service = new Intent(context, CounterService.class);
        context.stopService(service);
        context.startService(service);
        report("saved " + Settings.load(context));
    }

    /**
     * Log the outcome and hand it back to the sender as result data.
     *
     * @param message the outcome
     */
    private void report(String message) {
        Log.i(TAG, message);
        setResultData(message);
    }
}
