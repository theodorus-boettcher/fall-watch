package com.doorcounter.app;

import android.Manifest;
import android.app.admin.DevicePolicyManager;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

/**
 * Takes provisioning commands from the laptop while this app is the device owner.
 *
 * <p>The manifest only lets callers holding {@code android.permission.DUMP} reach it: the adb
 * shell has that permission, normal apps do not. Send a command with
 * {@code adb shell am broadcast -n com.doorcounter.app/.ProvisionReceiver --es op <op>
 * [--esa packages a,b,c]}. Ops: {@code grant-camera}, {@code hide}, {@code unhide},
 * {@code release} (give up the device-owner and admin roles, so the app can be uninstalled; works
 * without device owner too). The outcome comes back as the broadcast's result data, which
 * {@code am broadcast} prints.
 */
public final class ProvisionReceiver extends BroadcastReceiver {
    private static final String TAG = "DoorProvision";

    @Override
    public void onReceive(Context context, Intent intent) {
        DevicePolicyManager dpm = (DevicePolicyManager) context.getSystemService(Context.DEVICE_POLICY_SERVICE);
        String ownPackage = context.getPackageName();
        ComponentName admin = new ComponentName(context, AdminReceiver.class);
        String op = intent.getStringExtra("op");
        if ("release".equals(op)) {
            // Android 6 has no "dpm remove-active-admin", so the app drops its own admin role:
            // without this the app cannot be uninstalled after giving up device owner.
            if (dpm.isDeviceOwnerApp(ownPackage)) {
                dpm.clearDeviceOwnerApp(ownPackage);
            }
            dpm.removeActiveAdmin(admin);
            report("released");
            return;
        }
        if (!dpm.isDeviceOwnerApp(ownPackage)) {
            report("error: not device owner");
            return;
        }
        String[] packages = intent.getStringArrayExtra("packages");
        if ("grant-camera".equals(op)) {
            boolean granted = dpm.setPermissionGrantState(admin, ownPackage, Manifest.permission.CAMERA,
                    DevicePolicyManager.PERMISSION_GRANT_STATE_GRANTED);
            report("camera_granted=" + granted);
        } else if ("hide".equals(op)) {
            report(setHidden(dpm, admin, packages, true));
        } else if ("unhide".equals(op)) {
            report(setHidden(dpm, admin, packages, false));
        } else {
            report("error: unknown op " + op);
        }
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

    /**
     * Hide or unhide packages. A hidden package stays installed but cannot run.
     *
     * @param dpm the device policy manager
     * @param admin this app's admin component
     * @param packages package names, may be null if the extra was missing
     * @param hidden true to hide, false to unhide
     * @return a one-line summary: how many changed, and the names that did not
     */
    private static String setHidden(DevicePolicyManager dpm, ComponentName admin, String[] packages, boolean hidden) {
        if (packages == null) {
            return "error: no packages";
        }
        int changed = 0;
        StringBuilder failed = new StringBuilder();
        for (String pkg : packages) {
            boolean ok;
            try {
                ok = dpm.setApplicationHidden(admin, pkg.trim(), hidden);
            } catch (SecurityException e) {
                ok = false;
            }
            if (ok) {
                changed++;
            } else {
                failed.append(' ').append(pkg.trim());
            }
        }
        return (hidden ? "hidden=" : "unhidden=") + changed + " failed=" + (packages.length - changed) + ":" + failed;
    }
}
