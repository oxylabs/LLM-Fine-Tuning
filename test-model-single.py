import os, copy

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else torch.float16

generator = pipeline(
    "text-generation",
    model="./Amazon-title-gen/final",
    device=device,
    dtype=dtype
)

gen_config = copy.deepcopy(generator.model.generation_config)
gen_config.max_new_tokens = 120
gen_config.do_sample = False

input = "10\" x 12\" heavy duty wet cleaning wipes. The strongest and largest wet paper towel cleaning wipe. No water required. Removes heavy stains others won't including: grease, tar, ink, paint, permanent marker, wax, scuffs, lip stick, nail polish, food and drinks, pet stains and more. These moist towels are gentle on hands and skin. Wipes contain aloe, Vitamin E and lanolin to protect your hands and leave them clean and soft. Our heavy duty cleaning wipes do not contain disinfecting or antibacterial properties. Tub O' Towels can be used as automotive and car cleaning wipes, for the office, boating, and RV's, and around the home. Use these tough cleaning wipes on Fabric and Carpet, Leather, Vinyl, Metal, Counter Tops, Walls, For Cleaning Appliances, Tile, Cabinets, Toilets, Tubs and more. Tub O' Towels 90-Count Dual Texture 75-Count Tub O' Towels 40-Count Stainless Steel Wipes Granite & Marble Wipes Carpet & Upholstery Wipes Tub O' Towels Spray. Brand name: Tub O' Towels; Container type: Canister; Included components: cleaning wipes; Item form: Wipes; Item weight: 1.3 pounds; Manufacturer: Federal Process; Material type free: Water Free; Number of items: 1; Package size name: 90-COUNT; Scent: Citrus; Sheet count: 90; Surface recommendation: Fabric, Carpet, Metal, Wall, Tile, Toilet; Unit count: 90 Count" + "\n"
 
output = generator(
    input,
    generation_config=gen_config,
    return_full_text=False,
    clean_up_tokenization_spaces=False
)

print(f"\n Output:\n {output[0]['generated_text']}")