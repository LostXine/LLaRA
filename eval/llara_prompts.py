action_prompts = [
        "Could you write down what needs to be done to complete the task on this scene?",
        "List out the actions needed to accomplish the task in this scene.",
        "What actions are necessary to perform the task on this scene?",
        "Can you describe what needs to be done on this scene to complete the task?",
        "What steps are required to perform the task shown in this scene?",
        "List the actions needed to perform the task given below.",
        "On the following scene, could you list what actions are required to perform the task?",
        "Describe what actions are needed on this scene to complete the task.",
        "What do you need to do on this scene to accomplish the task?",
        "List the actions required to perform the task given on this scene.",
        "Could you please describe the steps needed to perform the task on this scene?",
        "Write down the actions required to perform the task on this scene.",
        "Please write down the actions required to perform the task shown below.",
        "Can you explain what needs to be done to perform the task in this scene?",
        "Describe the actions required to complete the task on this scene.",
        ]


detection_prompts = [
    "Identify and describe each object in the image. For each object, list it in the format <b>(x, y), {w, h}</b>, where x and y represent the coordinates of the bounding box center, and w and h represent the width and height of the bounding box. The image coordinates should start from the top left corner and be normalized between 0 and 1.",
    "Catalog all the objects present in the image. For every object, use the format <b>(x, y), {w, h}</b>, with x and y indicating the center of the object's bounding box coordinates, and w and h specifying the width and height. The coordinates are normalized from the top left corner, ranging from 0 to 1.",
    "List each object in the image and describe it. Use the format <b>(x, y), {w, h}</b> for each object, where x and y denote the center coordinates of the bounding box, and w and h are the width and height of the bounding box. The coordinates should start from the top left corner and be normalized to a scale of 0 to 1.",
    "Provide descriptions for all objects within the image. Each object should be listed using the format <b>(x, y), {w, h}</b>, where x and y are the coordinates of the bounding box center, and w and h are the width and height. The coordinates should be normalized, starting from the top left corner, within a range of 0 to 1.",
    "Enumerate and describe every object found in the image. For each object, utilize the format <b>(x, y), {w, h}</b>, where x, y are the bounding box center coordinates and w, h are the dimensions (width and height) of the bounding box. The coordinates begin at the top left corner and are normalized between 0 and 1.",
    "Detail all the objects within the image, listing each one using the format <b>(x, y), {w, h}</b>. Here, x and y represent the coordinates of the bounding box center, while w and h indicate the width and height. The coordinates start from the top left corner and are normalized to the range of 0 to 1.",
    "Document each object present in the image. For each object, use the format <b>(x, y), {w, h}</b>, where x and y are the coordinates of the center of the bounding box, and w and h are the width and height. The coordinates should be normalized, starting from the top left corner, and range from 0 to 1.",
    "For each object in the image, provide a description using the format <b>(x, y), {w, h}</b>. Here, x and y denote the coordinates of the bounding box center, and w and h represent the width and height of the bounding box. The coordinates are normalized to a scale of 0 to 1, starting from the top left corner.",
    "Describe all the objects seen in the image, and list them using the format <b>(x, y), {w, h}</b>. The x and y values are the coordinates for the center of the bounding box, while w and h represent its width and height. The coordinates should be normalized from the top left corner, within a range of 0 to 1.",
    "Identify and list each object found in the image. For each one, use the format <b>(x, y), {w, h}</b>. In this format, x and y are the coordinates for the bounding box center, and w and h are the width and height. The coordinates are to be normalized starting from the top left corner, ranging from 0 to 1.",
    "List and describe each object in the image using the format <b>(x, y), {w, h}</b>. Here, x and y correspond to the coordinates of the bounding box center, and w and h specify the width and height of the bounding box. The coordinates should start from the top left corner and be normalized to the range of 0 to 1.",
    "Provide a description for each object in the image, formatted as <b>(x, y), {w, h}</b>. The x and y values indicate the center coordinates of the bounding box, while w and h represent the width and height. The coordinates start from the top left corner and are normalized between 0 and 1.",
    "Catalog each object within the image, using the format <b>(x, y), {w, h}</b> for each one. In this format, x and y are the coordinates for the center of the bounding box, and w and h are the width and height. The coordinates should be normalized, beginning at the top left corner and ranging from 0 to 1.",
    "Enumerate all the objects in the image, providing descriptions for each using the format <b>(x, y), {w, h}</b>. The x and y values represent the center coordinates of the bounding box, while w and h indicate its width and height. The coordinates are normalized starting from the top left corner, within a range of 0 to 1.",
    "Describe each object in the image, listing them in the format <b>(x, y), {w, h}</b>. Here, x and y denote the center coordinates of the bounding box, and w and h specify the width and height. The coordinates should be normalized from the top left corner, ranging from 0 to 1."
]


localization_prompts = [
    "Where is {object} located in the image? Please use the format <b>(x, y), {w, h}</b> where x and y represent the center coordinates of the bounding box, and w and h are the width and height. The coordinates start from the top left corner and are normalized to a scale of 0 to 1.",
    "Can you provide the location of {object} in the image? Format it as <b>(x, y), {w, h}</b>, with x and y as the center coordinates of the bounding box and w and h as the width and height. The coordinates should begin at the top left corner and be normalized from 0 to 1.",
    "What are the coordinates of {object} in the image? Use the format <b>(x, y), {w, h}</b>, where x and y are the center of the bounding box, and w and h represent the width and height. Coordinates should start at the top left corner and be normalized to a range of 0 to 1.",
    "Please specify the location of {object} in the image. List it in the format <b>(x, y), {w, h}</b>, where x and y denote the bounding box center coordinates, and w and h are the width and height. The coordinates begin from the top left corner and should be normalized to 0 to 1.",
    "What is the position of {object} within the image? Use the format <b>(x, y), {w, h}</b> to describe it, with x and y as the center coordinates of the bounding box, and w and h as the width and height. The coordinates start at the top left corner and are normalized to a scale of 0 to 1.",
    "Describe the location of {object} in the image using the format <b>(x, y), {w, h}</b>. In this format, x and y denote the center coordinates of the bounding box, while w and h represent its width and height. Coordinates should be normalized from the top left corner, ranging from 0 to 1.",
    "Can you detail the location of {object} in the image? Format it as <b>(x, y), {w, h}</b>, where x and y indicate the bounding box center, and w and h represent the width and height. The coordinates should be normalized to a scale of 0 to 1 starting from the top left corner.",
    "Provide the location of {object} in the image using the format <b>(x, y), {w, h}</b>. Here, x and y are the center coordinates of the bounding box, and w and h are the width and height. The coordinates begin at the top left corner and are normalized from 0 to 1.",
    "Where is {object} positioned in the image? Use the format <b>(x, y), {w, h}</b>, where x and y denote the center coordinates of the bounding box, and w and h are the width and height. The coordinates should be normalized to a range of 0 to 1 starting from the top left corner.",
    "Specify the location of {object} in the image in the format <b>(x, y), {w, h}</b>. In this format, x and y represent the bounding box center, and w and h are the width and height. The coordinates should start from the top left corner and be normalized between 0 and 1.",
    "What is the exact position of {object} in the image? Format the coordinates as <b>(x, y), {w, h}</b>, where x and y are the center of the bounding box and w and h denote its width and height. The coordinates start from the top left corner and are normalized to a scale of 0 to 1.",
    "Describe where {object} is located in the image using the format <b>(x, y), {w, h}</b>. Here, x and y indicate the bounding box center coordinates, and w and h specify its width and height. The coordinates should be normalized starting from the top left corner, within the range of 0 to 1.",
    "Could you tell me the location of {object} in the image? Use the format <b>(x, y), {w, h}</b>, where x and y denote the center of the bounding box and w and h are the width and height. Coordinates start at the top left corner and should be normalized between 0 and 1.",
    "Provide the coordinates of {object} in the image in the format <b>(x, y), {w, h}</b>. Here, x and y are the center of the bounding box, while w and h represent its width and height. The coordinates should start from the top left corner and be normalized to 0 to 1.",
    "How is the {object} located in the image? List its coordinates using the format <b>(x, y), {w, h}</b>, where x and y are the center coordinates of the bounding box, and w and h indicate its width and height. The coordinates begin at the top left corner and are normalized to a range of 0 to 1."
]