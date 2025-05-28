from openvino.runtime import Core

core = Core()
model = core.read_model("D:/GhostTrack-Powered By VINOFlow/GhostTrack/best.xml")
compiled_model = core.compile_model(model, "CPU")

output_layer = compiled_model.output(0)
print("Output shape:", output_layer.shape)
