import matplotlib.pyplot as plt

def figsave(path,title,imagesize=(19.2, 10.44)):
    fig = plt.gcf()
    fig.set_size_inches(imagesize)
    # fig.tight_layout()
    save_path = path + title+'.png'
    # save_path = "/Users/Dou/Documents/Astro-Research/2017_fall/2017.10.11/" + "actual_1011_19.png"
    fig.savefig(save_path, dpi=200)
    plt.close()