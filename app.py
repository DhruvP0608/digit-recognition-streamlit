import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.set_page_config(
    page_title="Digit Recognition AI",
    layout="wide"
)

# ===== CLEAN UI CSS =====

st.markdown("""
<style>

.title{
font-size:40px;
font-weight:700;
color:#4CAF50;
margin-bottom:10px;
}

.card{
padding:18px;
border-radius:10px;
border:1px solid #2a2a2a;
background-color:#0E1117;
}

.metric{
font-size:28px;
font-weight:600;
color:#4CAF50;
}

.label{
font-size:14px;
color:gray;
}

.section{
padding:20px;
border-radius:10px;
border:1px solid #2a2a2a;
margin-top:20px;
}

</style>
""",unsafe_allow_html=True)


model=tf.keras.models.load_model("digit_model.h5")

train_acc=np.load("train_acc.npy")
val_acc=np.load("val_acc.npy")

train_loss=np.load("train_loss.npy")
val_loss=np.load("val_loss.npy")


st.markdown(
'<div class="title">AI Handwritten Digit Recognition System</div>',
unsafe_allow_html=True
)

tab1,tab2=st.tabs(["Prediction","Model Dashboard"])


# =====================
# PREPROCESS FUNCTION
# =====================

def preprocess_image(img):

    img=cv2.GaussianBlur(img,(5,5),0)

    _,img=cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
    )

    kernel=np.ones((3,3),np.uint8)

    img=cv2.morphologyEx(
        img,
        cv2.MORPH_OPEN,
        kernel
    )

    img=cv2.dilate(
        img,
        kernel,
        iterations=1
    )

    contours,_=cv2.findContours(
        img.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours)>0:

        biggest=max(contours,key=cv2.contourArea)

        x,y,w,h=cv2.boundingRect(biggest)

        digit=img[y:y+h,x:x+w]

    else:

        digit=img

    h,w=digit.shape

    size=max(h,w)

    square=np.zeros((size,size),dtype=np.uint8)

    y_offset=(size-h)//2
    x_offset=(size-w)//2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ]=digit

    digit=cv2.resize(square,(20,20))

    processed=np.pad(
        digit,
        ((4,4),(4,4)),
        "constant"
    )

    coords=np.column_stack(
        np.where(processed>0)
    )

    if len(coords)>0:

        center=coords.mean(axis=0)

        shift_y=int(14-center[0])
        shift_x=int(14-center[1])

        M=np.float32([
            [1,0,shift_x],
            [0,1,shift_y]
        ])

        processed=cv2.warpAffine(
            processed,
            M,
            (28,28)
        )

    return processed


# =====================
# SIDEBAR
# =====================

st.sidebar.title("Project Info")

st.sidebar.write("Model : CNN")

st.sidebar.write("Dataset : MNIST")

st.sidebar.write("Accuracy : 99%")

st.sidebar.write("Framework : TensorFlow")

st.sidebar.write("Interface : Streamlit")

st.sidebar.divider()

st.sidebar.write("Input : Handwritten digit image")

st.sidebar.write("Output : Predicted digit")


# =====================
# PREDICTION TAB
# =====================

with tab1:

    st.subheader("Digit Prediction")

    uploaded_file=st.file_uploader(
        "Upload handwritten digit",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file is not None:

        file_bytes=np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        img=cv2.imdecode(file_bytes,0)

        processed_img=preprocess_image(img)

        img_input=processed_img.astype("float32")/255.0

        img_input=img_input.reshape(1,28,28,1)

        prediction=model.predict(img_input)

        digit_pred=np.argmax(prediction)

        confidence=np.max(prediction)

        col1,col2=st.columns(2)

        with col1:

            st.image(
                uploaded_file,
                width=260,
                caption="Uploaded Image"
            )

            st.image(
                processed_img,
                width=160,
                caption="Processed Image",
                clamp=True
            )

        with col2:

            st.success(
                f"Predicted Digit : {digit_pred}"
            )

            st.write(
                f"Confidence : {confidence*100:.2f}%"
            )

            st.progress(float(confidence))

            if confidence<0.75:

                st.warning(
                "Low confidence prediction"
                )


# =====================
# DASHBOARD TAB
# =====================

with tab2:

    st.subheader("Model Performance")

    col1,col2,col3,col4=st.columns(4)

    with col1:

        st.markdown("""
        <div class="card">
        <div class="label">Training Accuracy</div>
        <div class="metric">%.2f%%</div>
        </div>
        """%(train_acc[-1]*100),
        unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">
        <div class="label">Validation Accuracy</div>
        <div class="metric">%.2f%%</div>
        </div>
        """%(val_acc[-1]*100),
        unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card">
        <div class="label">Final Loss</div>
        <div class="metric">%.4f</div>
        </div>
        """%(val_loss[-1]),
        unsafe_allow_html=True)

    with col4:

        st.markdown("""
        <div class="card">
        <div class="label">Epochs</div>
        <div class="metric">%d</div>
        </div>
        """%(len(train_acc)),
        unsafe_allow_html=True)

    st.divider()

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Accuracy Curve")

        fig1=plt.figure()

        plt.plot(train_acc)
        plt.plot(val_acc)

        plt.legend(["Training","Validation"])

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.title("Accuracy")

        st.pyplot(fig1)

    with col2:

        st.subheader("Loss Curve")

        fig2=plt.figure()

        plt.plot(train_loss)
        plt.plot(val_loss)

        plt.legend(["Training","Validation"])

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Loss")

        st.pyplot(fig2)

    st.divider()

    st.subheader("Model Summary")

    st.write("CNN with 2 convolution layers and dense classifier")

    st.write("Optimizer : Adam")

    st.write("Loss : Sparse categorical crossentropy")

    st.write("Training samples : 60000")

    st.write("Testing samples : 10000")