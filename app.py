import gradio as gr
import subprocess
import os,sys,asyncio,warnings
#import spaces
import torch

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore", module="torchvision.io._video_deprecation_warning")

extensions_dir = "./torch_extension/"
os.environ["TORCH_EXTENSIONS_DIR"] = extensions_dir

from networks.generator import Generator

device = torch.device("cuda")
gen = Generator(size=512, motion_dim=40, scale=2).eval().to(device).to(torch.float16)
ckpt_path = "../../../LIAX-release/model/lia-x.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = 'model/lia-x.pt'
    assert os.path.exists(ckpt_path)
gen.load_state_dict(torch.load(ckpt_path, weights_only=True))
#gen.eval().to(torch.float16) 
torch.cuda.empty_cache()

chunk_size=8 # number of frames to be generated at the same time

if 1:
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if total_memory < 8: chunk_size = 2
    elif total_memory < 12: chunk_size = 4
    elif total_memory < 16: chunk_size = 6
    print ('Adaptive chunk size:',chunk_size)

def load_file(path):

    with open(path, 'r', encoding='utf-8') as f:
        content  = f.read()

    return content

custom_css = """
<style>
  body {
    font-family: Georgia, serif; /* Change to your desired font */
  }
  h1 {
    color: black; /* Change title color */
  }
</style>
"""


with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML(load_file("assets/title.md"))
    with gr.Row():
        with gr.Accordion(open=False, label="Instruction"):
            gr.Markdown(load_file("assets/instruction.md"))
    
    with gr.Row():
        with gr.Tabs():
            from gradio_tabs.animation import animation
            from gradio_tabs.vid_edit import vid_edit
            from gradio_tabs.img_edit import img_edit
            animation(gen, chunk_size, device)
            img_edit(gen, device)
            vid_edit(gen, chunk_size, device)

    
demo.launch(
    #server_name='0.0.0.0',
    #server_port=10006,
    allowed_paths=["./data/source","./data/driving"]
)
