from gradio_client import Client

client = Client("https://6041351f74f0dabcb0.gradio.live/")
result = client.predict(
				"medium",
				"Spanish",
				"",	
				["test.wav"],	# List[str] (List of filepath(s) or URL(s) to files) in 'Upload Files' File component
				"",#Microphone Input' Audio component
				"transcribe",	
				"none",	
				5,	
				5,	
				False,
				False,
				api_name="/predict"
)
print(result[2])