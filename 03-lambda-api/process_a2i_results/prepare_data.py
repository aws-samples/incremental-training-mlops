object_categories = ["Barking", "Howling", "Crying", "COSmoke","GlassBreaking","Other"]
object_categories_dict = {str(j): i for i, j in enumerate(object_categories)}

def convert_a2i_to_augmented_manifest(a2i_output):
    
    label = a2i_output['humanAnswers'][0]['answerContent']['sentiment']['label']
    s3_path = a2i_output['inputContent']['taskObject']
    filename = s3_path.split('/')[-1][:-4]
    label_id = str(object_categories_dict[label]) 
    return '{},{},{}'.format(filename, label_id, label), s3_path