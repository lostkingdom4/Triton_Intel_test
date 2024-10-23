from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers import ModuleReplacer
import torch
from transformers import GPTJModel,GPTJConfig
from transformers.models.gptj.modeling_gptj import GPTJBlock
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules._transformer_configuration import ImplementMode


def print_layer_composition(module, indent=0):
    """Recursively print each module and its submodules."""
    # Print the class name of the current module
    print("  " * indent + f"{module.__class__.__name__}:")

    # Loop through the submodules of the current module
    for name, sub_module in module.named_children():
        print("  " * (indent + 1) + f"Submodule '{name}':")
        print_layer_composition(sub_module, indent + 2)

with torch.no_grad():
    module_dict = ModuleReplacer.default_replaced_module_dict()
    layer_dict = ModuleReplacer.default_replaced_layer_dict()
    fn_dict = ModuleReplacer.default_override_function_list()

    for old,new in module_dict.items():
        print(old)
        print(new)
        # print_layer_composition(old)
        # traced_cell = torch.jit.trace(module)

        # exported_program: torch.export.ExportedProgram = export(
        #     Mod(), args=example_args
        # )

        model_name = "EleutherAI/gpt-j-6B"
        # model = GPTJModel.from_pretrained(model_name)

        # Get model configuration to match embedding size
        config = GPTJConfig.from_pretrained(model_name)
        device = 'xpu'

        # Generate a reasonable input tensor for the GPTJBlock
        # Assume batch size = 1, sequence length = 10, and embedding size = config.hidden_size (which is 4096)
        batch_size = 4
        sequence_length = 1024
        embedding_size = config.hidden_size


        # Pass the input through the GPTJBlock
        # module = new(old,config).to(device)
        module = old(config).to(device)
        # module = module.half()
        input_embeddings = torch.randn(batch_size, sequence_length, embedding_size).to(device)
        attention_mask = torch.ones(batch_size, 1, 1, sequence_length).to(device)
        position_ids = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1).to(device)
        use_cache = True
        output_attentions = True

        print(input_embeddings.shape)
        print(attention_mask.shape)
        print(position_ids.dtype)
        # exit()
        
        def forward_wrapper(hidden_states, attention_mask, position_ids):
            # Call the original forward function of the module with tensor inputs
            return module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,                # Hardcoded or default value for tracing
                output_attentions=True         # Hardcoded or default value for tracing
            )
        for name, param in module.named_parameters():
             with torch.no_grad():
                param.copy_(param.detach())
                param.requires_grad=False
        # exit()
        output = module(input_embeddings, 
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=use_cache,
                        output_attentions=output_attentions)
        print(len(output))
        print(type(output))      
        traced_module = torch.jit.trace(forward_wrapper, (input_embeddings, attention_mask, position_ids))
        print(traced_module.graph)

        enable_naive_path = False

        impl_mode = (
            ImplementMode.naive if enable_naive_path else ImplementMode.optimized
        )

        decoder_kwargs={}
    
        new_module = new(
            module.half(),
            config,
            dtype='fp16',
            device="xpu",
            module_name="",
            impl_mode=impl_mode,
            tp_size=1,
            tp_group=None,
            **decoder_kwargs,
        )

        def forward_wrapper_2(hidden_states, attention_mask, position_ids):
            # Call the original forward function of the module with tensor inputs
            return new_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,                # Hardcoded or default value for tracing
                output_attentions=True         # Hardcoded or default value for tracing
            )
        
        print(new_module.parameters())  # Should show torch.float16
        input_embeddings = input_embeddings.half()
        attention_mask = attention_mask.half()
        # TODO: Double check with run_generation
        # position_ids = position_ids.half()
        print(input_embeddings.dtype)
        print(attention_mask.dtype)
        print(position_ids.dtype)

        for param in new_module.parameters():
            print(param.dtype)

        output = new_module(input_embeddings, attention_mask=attention_mask,position_ids=position_ids,use_cache=use_cache,output_attentions=output_attentions)
        # print(len(output))
        # print(type(output))
        # for out in output:
        #     print(out)
        # traced_module = torch.jit.script(new_module())
        # print(traced_module.graph)
        
        new_traced_module = torch.jit.trace(forward_wrapper_2, (input_embeddings, attention_mask, position_ids))
        print(new_traced_module.graph)

        print(new_traced_module==traced_module)

        # exported_program: torch.export.ExportedProgram = export(
        #     model, args=(data,), kwargs={}
        # )
        # print(traced_cell.code)
        break
        module()
