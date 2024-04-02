import azure.ai.vision as sdk
import os

class ImageAnalyzer:
    def __init__(self):
        self.service_options = sdk.VisionServiceOptions(os.getenv('VISION_URL'), os.getenv('VISION_API'))
        self.analysis_options = sdk.ImageAnalysisOptions()
        self.analysis_options.features = (
            sdk.ImageAnalysisFeature.DENSE_CAPTIONS |
            sdk.ImageAnalysisFeature.TAGS |
            sdk.ImageAnalysisFeature.OBJECTS# |
           # sdk.ImageAnalysisFeature.CAPTION
        )

    def Analyzed(self, sourse, threshold=0.7 ,source_file=True):
        if source_file:
            image_source_buffer = sdk.ImageSourceBuffer()
            image_source_buffer.image_writer.write(sourse)
            vision_source = sdk.VisionSource(frame_source=image_source_buffer)
        else:
            vision_source = sdk.VisionSource(url=sourse)
        image_analyzer = sdk.ImageAnalyzer(self.service_options, vision_source, self.analysis_options)
        result = image_analyzer.analyze()

        captions = []
        tags = []
        objs=[]
        caption=''
        if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

            if result.dense_captions is not None:
                for caption in result.dense_captions:
                    if caption.confidence > threshold and caption.content not in captions:
                        captions.append(caption.content)
            if len(captions)==0:
                    captions=['']

            if result.tags is not None:
                for tag in result.tags:
                    if tag.confidence > threshold and tag.name not in tags:
                        tags.append(tag.name)
            if len(tags)==0:
                    tags=['']

            if result.objects is not None:
                for obj in result.objects:
                    if obj.confidence > threshold and obj.name not in objs:
                        objs.append(obj.name)
            if len(objs)==0:
                    objs=['']

        return captions,tags,objs
    
if __name__=='__main__':
     ImageAnalyzer