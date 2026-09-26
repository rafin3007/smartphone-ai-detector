                        camera_image,
                        model_bundle,
                        top_k=top_k,
                    )

                camera_prediction_result = render_prediction(
                    predictions,
                    component_dataframe,
                    component_counts,
                    phone_specifications,
                    valuable_metals_data,
                    confidence_threshold,
                    image=camera_image,
                )

                if (
                    camera_prediction_result
                    and camera_prediction_result.get(
                        "damage_result"
                    )
                    is not None
                ):
                    render_graphical_valuation_demo(
                        image=camera_image,
                        top_prediction=camera_prediction_result[
                            "top"
                        ],
                        damage_result=camera_prediction_result[
                            "damage_result"
                        ],
                        pricing_data=pricing_data,
                        component_dataframe=component_dataframe,
                        component_counts=component_counts,
                        metals_data=valuable_metals_data,
                        internal_layouts=internal_layouts,
                        component_value_config=component_value_config,
                    )
            else:
                st.info(
                    "Allow camera access and "
                    "take a picture."
                )

with mode[2]:
    if not model_ready:
        st.error("AI detector is unavailable. Use the Manual valuation assistant tab instead.")
    else:
        st.subheader("Real-time smartphone recognition")
        st.caption(
            "Press START, allow browser camera access, and point the camera at "
            "the phone. The prediction is drawn on the video."
        )

        context = webrtc_streamer(
            key="smartphone-live-detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SmartphoneVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 960},
                    "height": {"ideal": 720},
                    "facingMode": "environment",
                },
                "audio": False,
            },
            async_processing=True,
        )

        if context.video_processor:
            context.video_processor.configure(
                model_bundle=model_bundle,
                threshold=confidence_threshold,
                frame_interval=live_frame_interval,
            )

        st.info(
            "Live classification is frame-based. A lower inference interval is "
            "more responsive but uses more computing resources."
        )


with mode[3]:
    render_manual_valuation_assistant(
        pricing_data,
        component_dataframe,
        component_counts,
        phone_specifications,
        valuable_metals_data,
    )

with mode[4]:
    if not model_ready:
        st.info("The AI detector model is not loaded. Manual valuation remains available.")
    else:
        st.subheader("Loaded model")

        model_details = {
            "Architecture": model_bundle["model_name"],
            "Device": str(model_bundle["device"]),
            "Input size": model_bundle["image_size"],
            "Classes": len(model_bundle["class_names"]),
            "Normalization mean": model_bundle["mean"],
            "Normalization standard deviation": model_bundle["std"],
        }

        st.json(model_details)

        st.subheader("Recognized classes")

        class_table = pd.DataFrame(
            {
                "Training label": model_bundle["class_names"],
                "Display name": [
                    pretty_class_name(value)
                    for value in model_bundle["class_names"]
                ],
            }
        )

        st.dataframe(
            class_table,
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "The model can recognize only the classes present during training. "
