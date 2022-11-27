# Display simulation result
import glob
import os
from PIL import Image as pilImage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_gif(folders):
    from os import path
    frames = []
    for b in folders:
        try:
            i = 0
            while i > -1:
                frames += [pilImage.open(os.path.join(f"{b}", f"frame_{i}.png"))]
                i += 1
        except Exception as e:
            pass
    frame_one = frames[0]
    frame_one.save("example_output.gif", format="GIF", append_images=frames,
                   save_all=True, duration=3000 // len(frames), loop=0)

def make_video(folders):
    import imageio
    from os import path
    frames = []
    for b in folders:
        i = 0
        while True:
            frame_path = os.path.join(f"{b}", f"frame_{i}.png")
            if not path.exists(frame_path):
                break
            frames.append(frame_path)
            i += 1


    writer = imageio.get_writer('./video/testNoEpisode.mp4', fps=20)
    for im in frames:
        writer.append_data(imageio.imread(im))
    writer.close()

png = False
videos = glob.glob('./results/testNoEpisode/recording_*/')
videos.sort()
print(videos)

if png:
    make_gif(videos)
else:
    make_video(videos)



#
# from PIL import Image
# im = Image.open('example_output.gif')

#output = ipyImage(open('example_output.gif', 'rb').read())
#display(ipyImage(data=open('example_output.gif','rb').read(), format='gif'))
#fra mi tocca andare
#ma dove la salva che noncapisco




    # import cv2
    # import os
    # from os import path
    # image_folder = './results/recording_*/'
    # video_name = 'video.avi'
    #
    # images = []
    # for b in videos:
    #     i = 0
    #     while True:
    #         frame_path = os.path.join(f"{b}", f"frame_{i}.png")
    #         if not path.exists(frame_path):
    #             break
    #         images.append(frame_path)
    #         i += 1
    #
    # frame = cv2.imread(images[0])
    # height, width, layers = frame.shape
    #
    # video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    #
    # for image in images:
    #     video.write(cv2.imread(os.path.join(image_folder, image)))
    #
    # cv2.destroyAllWindows()
    # video.release()