import os
from configGen import create_config, write_config_to_file, create_mushra_page, create_generic_page, create_consent_page, create_volume_page, create_questionnaire_field, create_finish_page

def create_file_dict(folder_path):
    file_dict = {}
    for folder_name, subfolders, files in os.walk(folder_path):
        file_dict[folder_name] = files
    return file_dict

def create_mushra_test_pages(folder_path, pages=[]):
    file_dict = create_file_dict(folder_path)

    for subfolder, files in file_dict.items():
        if subfolder != folder_path:
            soundName = subfolder.split("/")[-1]
            subfolder = subfolder.replace("..", "configs")
            instruction = "Rate the brightness of each stimuli compared to the reference "
            refFile = [s for s in files if "+0.000dB" in s][0]
            files.remove(refFile)

            # Create a mushra test page using the sounds inside the subfolder
            id= soundName
            name= soundName
            content= instruction
            showWaveform= True
            enableLooping= True
            strict= False
            reference= os.path.join(subfolder, refFile)
            createAnchor35= False
            createAnchor70= False
            randomize= True
            showConditionNames= False
            stimuli= {"C"+str(index+1): os.path.join(subfolder, item) for index, item in enumerate(files)}
            switchBack= True

            pages.append(create_mushra_page(id, name, content, showWaveform, enableLooping, strict, reference, createAnchor35, createAnchor70, randomize, showConditionNames, stimuli, switchBack))
            print(f"Creating Mushra test page for subfolder: {subfolder}")
    
    return pages

def process_for_html(text):
    text = text.replace("\n", "<br>")
    text = text.replace("'", "&#39;")
    return text

def get_welcome_page():
    id= "generic1"
    name= "Brightness Perception Test"
    contentText= "\
            Welcome to our online survey about brightness perception and thank you very much for participating! \n \n \
            In this survey, we wish to understand the sensation of brightness in context of music. \n \
            Brightness is used to describe the timbre of a musical instrument, human voice and other music like sounds. \n \n \
            With our study, we'd like to find out how you perceive the brightness of small song snippets. \n \n \
            We will give you a reference song and then ask you to rate the brightness of 4 snippets with respect to the reference. \n \
            Please use headphones to participate in this survey. \
            The survey will take approximately 20 minutes. \n \n\
            Please press <b> Next </b> below to proceed to the instructions page \n \
            "
    content = process_for_html(contentText)
    return create_generic_page(id, name, content)

def get_instruction_page(folder_path):
    """
       Create an instructional MUSHRA test to explain the task to the user
    """
    file_dict = create_file_dict(folder_path)
    subfolder = folder_path
    soundName = subfolder.split("/")[-1]
    subfolder = subfolder.replace("..", "configs")
    files = file_dict[folder_path]
    refFile = [s for s in files if "+0.000dB" in s][0]
    files.remove(refFile)

    id= "instruction1"
    name= "Instructions Page"
    contentText= "\
            The test will have 16 test cases. Each case will have one reference sound and four possibly edited versions. \
            Rate each edited condition sound with respect to the reference. \n \n \
            Assign each condition to one of the five categories. \
            If the condition sound seems unchanged, leave the slider at 50 i.e. <b>Same Brightness</b>. \
            If the condition sound seems <b>More Bright</b>, <b>Most Bright</b>, <b>Less Bright</b>, or <b>Least Bright</b>, set the slider to any number within that category. \n\
            For example, <b>C2</b> here is brighter and <b>C3</b> is less bright compared to reference. \n \n\
            While all the operations on this page could be done using a mouse, there are helpful keyboard shortcuts. \
            Press <b>0</b> to play reference, <b>1</b> for condition 1, <b>2</b> for condition 2 etc. Press the same key again to pause.\
            Press <b>Shift + Up key</b> to increase the slider value by 10. Press <b>Shift + Down key</b> to decrease the slider value by 10. \n \n\
            Adjust the loop markers to loop a shorter section of the original sound. \n \
            Please press <b> Next </b> below to proceed to the test \n \
            "
    content= process_for_html(contentText)
    showWaveform= True
    enableLooping= True
    strict= False
    reference= os.path.join(subfolder, refFile)
    createAnchor35= False
    createAnchor70= False
    randomize= True
    showConditionNames= True
    stimuli= {"C"+str(index+1): os.path.join(subfolder, item) for index, item in enumerate(files)}
    switchBack= True

    return create_mushra_page(id, name, content, showWaveform, enableLooping, strict, reference, createAnchor35, createAnchor70, randomize, showConditionNames, stimuli, switchBack)

if __name__ == "__main__":
    folder_path = "../resources/audio/experiment/"
    pages = []
    pages.append(get_welcome_page())
    pages.append(get_instruction_page(os.path.join(folder_path, "HerbieHancock_Snippet_1")))
    # pages.append(create_consent_page("consent1", "Consent", "Please give your consent to participate in the test"))
    # pages.append(create_volume_page("volume1", "Volume", "Please set the volume", "configs/resources/audio/calibration/volume/pinknoise_minus60db.wav"))
    pages.extend(create_mushra_test_pages(folder_path))

    questionnaire = create_questionnaire_field(True, True, True, True, True)
    pages.append(create_finish_page("finish1", "Finish", "Thank you for participating in the test!", "Your results were sent. Goodbye and have a nice day", True, False, True, questionnaire))
    config = create_config("MPCBrightness", "mpcBrightness", 1024, True, True, "https://formspree.io/f/xkndnwpv", pages)
    write_config_to_file(config, "mpc_brightness.yaml")




    

