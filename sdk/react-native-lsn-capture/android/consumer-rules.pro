# Applied to the host app's R8 run. Nothing in the SDK is reflected on except
# the bridge module (RN's own rules already keep every NativeModule), and the
# activity is kept through the manifest. Kept explicitly anyway so a host that
# disables RN's rules still works; everything else may be shrunk / obfuscated.
-keep class com.loadshare.lsncapture.LsnCapturePackage { <init>(...); }
-keep class com.loadshare.lsncapture.LsnCaptureModule { *; }
-keep class com.loadshare.lsncapture.CaptureActivity { <init>(); }
