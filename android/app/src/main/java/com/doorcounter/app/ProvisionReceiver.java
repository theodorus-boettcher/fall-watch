package com.doorcounter.app;

import android.Manifest;
import android.app.admin.DevicePolicyManager;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.net.wifi.WifiConfiguration;
import android.net.wifi.WifiManager;
import android.util.Log;

/**
 * Takes provisioning commands from the laptop while this app is the device owner.
 *
 * <p>The manifest only lets callers holding {@code android.permission.DUMP} reach it: the adb
 * shell has that permission, normal apps do not. Send a command with
 * {@code adb shell am broadcast -n com.doorcounter.app/.ProvisionReceiver --es op <op>
 * [--esa packages a,b,c]}. Ops: {@code grant-camera}, {@code hide}, {@code unhide},
 * {@code pin-wifi} ({@code --es ssid <name> --es bssid <access point, or any to undo>}),
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
        } else if ("pin-wifi".equals(op)) {
            report(pinWifi(context, intent.getStringExtra("ssid"), intent.getStringExtra("bssid")));
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
     * Lock a saved network to one access point, so the phone stops trying to switch between a router's
     * 5 GHz and 2.4 GHz radios (which share one name) and dropping off for minutes when a switch fails.
     * The saved password is left as it is. As device owner the app may change a network saved from the
     * Settings app.
     *
     * @param context the receiver's context
     * @param ssid the saved network's name
     * @param bssid the access point's address, e.g. {@code aa:bb:cc:dd:ee:ff}, or {@code any} to undo
     * @return a one-line outcome
     */
    private static String pinWifi(Context context, String ssid, String bssid) {
        if (ssid == null || bssid == null) {
            return "error: pin-wifi needs ssid and bssid";
        }
        WifiManager wifi = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        for (WifiConfiguration config : wifi.getConfiguredNetworks()) {
            if (!("\"" + ssid + "\"").equals(config.SSID)) {
                continue;
            }
            config.BSSID = bssid;
            config.preSharedKey = null;  // null leaves the saved password untouched
            int networkId = wifi.updateNetwork(config);
            if (networkId < 0) {
                return "error: Android refused the change";
            }
            boolean isEnabled = wifi.enableNetwork(networkId, true);
            return "pinned network_id=" + networkId + " bssid=" + bssid + " enabled=" + isEnabled
                    + " saved=" + wifi.saveConfiguration();
        }
        return "error: no saved network named " + ssid;
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
