from ipywidgets import FloatSlider, interact

humour = FloatSlider(min=-15, max=15, step=3, value=0)
pose = FloatSlider(min=-15, max=15, step=3, value=0)

@interact(pose=pose, humour=humour)
def do_thumb(humour, pose):
    z_sample = np.array([[humour, pose]]) * noise_std
    x_decoded = generator.predict(z_sample)
    face = x_decoded[0].reshape(img_rows, img_cols)
    plt.figure(figsize=(11.5, 11.5))
    ax = plt.subplot(111)
    ax.imshow(face)
    plt.axis("off")