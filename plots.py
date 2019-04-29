import matplotlib.pyplot as plt

def figsave(path,title,imagesize=(19.2, 10.44)):
    fig = plt.gcf()
    fig.set_size_inches(imagesize)
    # fig.tight_layout()
    save_path = path + title+'.png'
    fig.savefig(save_path, dpi=200)
    plt.close()