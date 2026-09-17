"""
Deployment Recommendations & The Three Decoupled Scientific Conclusions.
"""

import streamlit as st


def render_recommendations_page():
    st.markdown("### 🎯 Deployment Decision & Thesis Recommendations")
    st.markdown("""
    A standard pitfall in multimodal machine learning is asking the overly simplistic question: *"Which model has the highest mAP?"*
    In real-world robotics and autonomous systems, **accuracy without generalization is brittle, and accuracy without real-time throughput is fatal.**
    Therefore, our FYP research formally decouples the comparative synthesis into **three distinct, scientifically grounded conclusions**:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background-color:#ffffff;border:2px solid #2563eb;border-radius:8px;padding:20px;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div style="color:#1e40af;font-weight:800;font-size:13px;letter-spacing:1px;text-transform:uppercase;">Conclusion 1</div>
            <h3 style="margin:6px 0 10px 0;color:#0f172a;font-weight:800;">Peak Target Accuracy</h3>
            <div style="background-color:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;padding:5px 10px;border-radius:6px;font-weight:700;display:inline-block;margin-bottom:12px;font-size:12px;">
                WINNER: Stage 6 (YOLO11s)
            </div>
            <p style="font-size:13.5px;color:#1e293b;line-height:1.6;">
                For the safety-critical classes representing <b style="color:#0f172a;">91.8% of real-world traffic objects</b>, Stage 6 establishes the highest detection precision:
            </p>
            <ul style="font-size:13px;color:#334155;padding-left:18px;line-height:1.7;">
                <li><b style="color:#0f172a;">Pedestrians (People):</b> <span style="color:#15803d;font-weight:800;">77.49% mAP@50</span> (+9.4 pp over Stage 3, +44.3 pp over RGB)</li>
                <li><b style="color:#0f172a;">Vehicles (Car):</b> <span style="color:#15803d;font-weight:800;">85.26% mAP@50</span> (+2.0 pp over Stage 3)</li>
                <li><b style="color:#0f172a;">Validation Accuracy:</b> <span style="color:#15803d;font-weight:800;">81.08% mAP@50</span></li>
            </ul>
            <div style="font-size:12.5px;color:#475569;border-top:1px solid #f1f5f9;padding-top:10px;margin-top:12px;">
                <b style="color:#0f172a;">Best For:</b> High-precision perimeter defense, forensic surveillance, and offline intelligence analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color:#ffffff;border:2px solid #7c3aed;border-radius:8px;padding:20px;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div style="color:#6d28d9;font-weight:800;font-size:13px;letter-spacing:1px;text-transform:uppercase;">Conclusion 2</div>
            <h3 style="margin:6px 0 10px 0;color:#0f172a;font-weight:800;">Generalization Resilience</h3>
            <div style="background-color:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe;padding:5px 10px;border-radius:6px;font-weight:700;display:inline-block;margin-bottom:12px;font-size:12px;">
                WINNER: Stage 3 (Joint Fusion)
            </div>
            <p style="font-size:13.5px;color:#1e293b;line-height:1.6;">
                Under substantial empirical distribution shift (unseen nighttime routes, blinding glare, shadows), single sensors collapse while feature fusion endures:
            </p>
            <ul style="font-size:13px;color:#334155;padding-left:18px;line-height:1.7;">
                <li><b style="color:#0f172a;">Test Retention Rate:</b> <span style="color:#6d28d9;font-weight:800;">65.1% Retention</span> (vs. 38.4% for Direct Optical RGB)</li>
                <li><b style="color:#0f172a;">Unweighted 6-Class mAP:</b> <span style="color:#6d28d9;font-weight:800;">48.56%</span> (Highest project unweighted mean)</li>
                <li><b style="color:#0f172a;">Minority Robustness:</b> Bus 58.62%, Truck 14.04%</li>
            </ul>
            <div style="font-size:12.5px;color:#475569;border-top:1px solid #f1f5f9;padding-top:10px;margin-top:12px;">
                <b style="color:#0f172a;">Best For:</b> Environments with extreme, unpredictable day/night and adverse weather shifts.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color:#ffffff;border:2px solid #16a34a;border-radius:8px;padding:20px;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div style="color:#15803d;font-weight:800;font-size:13px;letter-spacing:1px;text-transform:uppercase;">Conclusion 3</div>
            <h3 style="margin:6px 0 10px 0;color:#0f172a;font-weight:800;">Real-Time Robotics & AVs</h3>
            <div style="background-color:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;padding:5px 10px;border-radius:6px;font-weight:700;display:inline-block;margin-bottom:12px;font-size:12px;">
                WINNER: Stage 5 (Late Fusion)
            </div>
            <p style="font-size:13.5px;color:#1e293b;line-height:1.6;">
                Autonomous platforms require deterministic latency ($\le 33\text{ ms} / \ge 30\text{ FPS}$) and hardware redundancy:
            </p>
            <ul style="font-size:13px;color:#334155;padding-left:18px;line-height:1.7;">
                <li><b style="color:#0f172a;">Real-Time Throughput:</b> <span style="color:#15803d;font-weight:800;">~51.5 FPS (19.41 ms)</span> (4x faster than TarDAL)</li>
                <li><b style="color:#0f172a;">Fault Tolerance:</b> <span style="color:#15803d;font-weight:800;">100% Graceful Degradation</span> under sensor loss</li>
                <li><b style="color:#0f172a;">No Generative Bottleneck:</b> Bypasses dense-block pixel synthesis</li>
            </ul>
            <div style="font-size:12.5px;color:#475569;border-top:1px solid #f1f5f9;padding-top:10px;margin-top:12px;">
                <b style="color:#0f172a;">Definitive Recommendation:</b> Safety-critical autonomous driving, robotic UGVs, and edge edge-embedded deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏛️ Master Architectural Decision Matrix (Summary)")
    st.markdown("""
| Architectural Dimension | Direct Optical (RGB) | Direct Thermal (IR) | Stage 3 Feature Fusion | Stage 5 Late Fusion (WBF) | Stage 6 Modern YOLO11s |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Detector Head** | YOLOv5su (9.12M) | YOLOv5su (9.12M) | YOLOv5su (9.12M) | Dual YOLOv5su (18.24M) | **YOLO11s (9.43M, C3k2+C2PSA)** |
| **Generative Overhead** | 0.0 ms | 0.0 ms | 66.43 ms | **0.0 ms (WBF: 0.58 ms)** | 61.61 ms |
| **Total Tesla T4 Latency** | 9.42 ms | 9.59 ms | 77.95 ms | **19.41 ms** 🏆 | 79.42 ms |
| **Inference Throughput** | 106.2 FPS | 104.3 FPS | 12.8 FPS | **51.5 FPS** 🏆 | 12.6 FPS |
| **Operational mAP@50 (conf=0.25)** | 25.59% | 23.54% | 40.12% | **30.29%** (+4.7% vs RGB) | **41.65%** 🏆 |
| **Pedestrian Score (People)** | 33.21% (BM) / 27.29% (Op) | 75.43% (BM) | 68.09% (BM) | 51.08% (Op) | **77.49%** 🏆 (Project Record) |
| **Vehicle Score (Car)** | 77.43% (BM) / 71.56% (Op) | 73.68% (BM) | 83.28% (BM) | 77.56% (Op) | **85.26%** 🏆 (Project Record) |
| **Sensor Loss Robustness** | Fails in dark/rain | Fails on cold objects | 0% (Single Failure) | **100% Graceful Degradation** 🏆 | 0% (Single Failure) |
| **Recommended Deployment** | High-contrast daylight | Thermal dark monitoring | Research benchmark | **Safety-Critical AVs / UGVs** 🏆 | **Offline Analytics & Forensics** 🏆 |
    """)
