import demo



def get_style_options():
    return demo.get_style_options()

def get_contained_class():
    # should only called after classify()
    return demo.get_contained_class()

def setup():
    # setup model and config
    demo.setup()

def set_input_image(input_image_path):
    # chould only called after setup()
    demo.set_input_image(input_image_path)

def classify():
    # chould only called after set_input_image()
    # classify input image, set up contained_classes
    demo.classify()

def set_style():
    # should only called after classify()
    # set the relation of class - style
    class_style_dict = {"background" : "none"}
    contained_classes = get_contained_class()
    for class_name in contained_classes:
        style = input(class_name + ":")
        class_style_dict[class_name] = style
    demo.set_style(class_style_dict)

def generate():
    # should only called after set_style()
    demo.generate()

def set_output_path(path = "output.png"):
    demo.set_output_path(path)

if __name__ == '__main__':

    setup()
    set_input_image(input("input file name:"))
    set_output_path(input("set output path:"))
    classify()

    # show available style options, reference style images in "styles" folder
    #
    style_options = get_style_options()
    print("available style options:")
    for style in style_options:
        print("    " + style)

    # show contained classes in the input image
    # should only called after classify()
    contained_classes = get_contained_class()
    print("contained classes:")
    for class_name in contained_classes:
        print("    " + class_name)

    set_style()
    generate()
