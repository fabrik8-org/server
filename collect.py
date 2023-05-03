from pymongo import MongoClient
from gridfs import GridFS
import base64

client = MongoClient('mongodb://localhost:27017/')

db = client['imagedb']
collection = db['images']


def collect_data(image_base64, prediction):
    # Decode the base64 image data
    # image_data = base64.b64decode(image_base64)

    # # Create a new GridFS object
    # fs = GridFS(db)

    # # Store the image data in GridFS and get its ObjectId
    # file_id = fs.put(image_data, filename=filename)

    # Save the ObjectId to your collection
    collection.insert_one({'image': image_base64, 'prediction': prediction})
