from gradio_client import Client

client = Client("https://14c5b40f5b6d6294b9.gradio.live/")
result = client.predict(
				"tiny",
				"Spanish",
				"",	
				["test.wav"],	# List[str] (List of filepath(s) or URL(s) to files) in 'Upload Files' File component
				"test.wav",	# str (filepath or URL to file) in 'Microphone Input' Audio component
				"transcribe",	
				"none",	
				5,	
				5,	
				True,
				True,	
				api_name="/predict"
)
print(result)