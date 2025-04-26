import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math
# helpers
topil = T.ToPILImage()

def show(img_tensor, label):
    params = {}
    is_gray = img_tensor.shape[0] == 1 # 1 -> gray / 3 -> rgb
    if is_gray:
        params["cmap"] = "gray"
    pil_img = topil(img_tensor)
    
    plt.imshow(pil_img, **params)
    plt.title(f"label: {label}")

def show_grid(img_tensors, title):
    num_items = len(img_tensors)
    auto_shape = ()
    for i in range(int(math.sqrt(num_items))+1)[::-1]:
        if num_items % i == 0:
            auto_shape = (i, num_items//i)
            print(i)
            break
    h, w = auto_shape
    fig, ax = plt.subplots(h, w)
    for i in range(h):
        for j in range(w):
            index = i*w+j
            img_tensor = img_tensors[index]
            image = ax[i, j].imshow(topil(img_tensor))
            # image.set_cmap("gray")
            ax[i, j].axis("off")
    fig.suptitle(title)
    return