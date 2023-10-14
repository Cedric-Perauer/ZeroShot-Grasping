from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import math
import json

app = Flask(__name__)

image_folder = '.'
images_per_page = 200

@app.route('/')
def index():
    global image_folder
    #image_folder = request.args.get('folder', default='images', type=str)
    page = request.args.get('page', default=1, type=int)
    start = (page - 1) * images_per_page
    end = start + images_per_page
    
    img_pth = '512_cnt_angle_orig/test_t/grasps_test2018/' 
    
    subfolders = os.listdir(img_pth)
    images_all = []
    print(subfolders)
    for subfolder in subfolders:
        objects = os.listdir(img_pth + subfolder + '/')
        for obj in objects:
            imgs = os.listdir(img_pth + subfolder + '/' + obj + '/')
            images_all.append(img_pth + subfolder + '/' + obj + '/' + imgs[0])
    
    images_all = images_all[:10]
    total_pages = math.ceil(len(images_all) / images_per_page)

    images_select = images_all[start:end]
    image_files = [img for img in images_select if img.endswith(('.png', '.jpg', '.jpeg'))]

    if os.path.exists('selected_images.json'):
        with open('selected_images.json', 'r') as f:
            selected_images = json.load(f)
    else:
        selected_images = []

    print(selected_images)
    marked_pages = []
    for i in range(total_pages):
        imgs_sub = set(images_all[i*images_per_page:(i+1)*images_per_page])
        if len(imgs_sub.intersection(set(selected_images))) > 0:
            marked_pages.append(i+1)

    return render_template('index.html', images=image_files,
                        selected_images=selected_images,
                        page=page,
                        total_pages=total_pages,
                        marked_pages=marked_pages
                        )


@app.route('/image/<path:filename>')
def serve_image(filename):
    print("/".join(filename.split('/')[:-1]))
    print(os.path.basename(filename.split('/')[-1]))
    return send_file(filename)
    #return send_from_directory("/".join(filename.split('/')[:-1]) + '/', os.path.basename(filename.split('/')[-1]))

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    print(data)
    selected_images = ["/".join(d.split('/')[4:]) for d in data]

    if os.path.exists('selected_images.json'):
        with open('selected_images.json', 'r') as f:
            selected_images_stored = json.load(f)
        selected_images.extend(selected_images_stored)

    with open('selected_images.json', 'w') as f:
        json.dump(selected_images, f)

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='.', help='Folder to display images from')
    args = parser.parse_args()
    app.run(debug=True)
