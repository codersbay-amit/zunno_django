from django.db import models

class BrandCreation(models.Model):
    logo = models.URLField(max_length=500, blank=True, null=True)  # Store the logo as a URL
    name = models.CharField(max_length=255)
    
    titleFont = models.CharField(max_length=100)
    subTitleFont = models.CharField(max_length=100)
    bodyFont = models.CharField(max_length=100)
    productImage= models.URLField(max_length=500, blank=True, null=True) 
    productPrompt=models.TextField(null=True)
    tagLine = models.CharField(max_length=255)
    industry = models.CharField(max_length=100)
    demographic = models.CharField(max_length=255)
    psychographic = models.CharField(max_length=255)
    audienceInterest = models.CharField(max_length=255)
    toneOfVoice = models.CharField(max_length=50)
    messageStyle = models.CharField(max_length=50)
    visualStyle = models.CharField(max_length=50)
    
    competitorBrands = models.JSONField(null=True)  # Store a list of competitors
    benchmarkBrands = models.JSONField()  # Store a list of benchmark brands
    brandGuidelineDocs = models.JSONField(default=list)  # Store a list of documents
    copyrightText = models.CharField(max_length=255)
    
    keywords = models.JSONField()  # Store a list of keywords
    ctaText = models.CharField(max_length=50)
    currentCampaign = models.CharField(max_length=255)
    
    visualElements = models.JSONField(default=list)  # Store visual elements

class Color(models.Model):
    brand = models.ForeignKey(BrandCreation, on_delete=models.CASCADE, related_name='colors')
    colorCode = models.CharField(max_length=50)  # e.g. "tomato" or "#FFFF00" for yellow
    type = models.CharField(max_length=50)  # e.g. "primary" or "secondary"

class Advanced(models.Model):
    brand = models.ForeignKey(BrandCreation, on_delete=models.CASCADE, related_name='advancedData')
    style = models.CharField(max_length=50)
    colorScheme = models.CharField(max_length=50)
    lighting = models.CharField(max_length=50)
    composition = models.CharField(max_length=50)
    perspective = models.CharField(max_length=50)
    mood = models.CharField(max_length=50)
    background = models.CharField(max_length=50)
    timePeriod = models.CharField(max_length=50)
def get_string(instance):
    # These parameters are being used to select the template
    return f"""
               {instance.name}, {instance.industry}, {instance.tagLine}, {instance.demographic}, {instance.psychographic}, {instance.messageStyle},
               {instance.competitorBrands}, {instance.benchmarkBrands}, {instance.keywords}, {instance.ctaText}, {instance.currentCampaign}
            """
