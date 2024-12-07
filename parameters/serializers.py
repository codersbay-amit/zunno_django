from rest_framework import serializers
from .models import BrandCreation, Color, Advanced

class ColorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Color
        fields = ['id', 'colorCode', 'type']

class AdvancedDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = Advanced
        fields = ['id', 'style', 'colorScheme', 'lighting', 'composition', 'perspective', 'mood', 'background', 'timePeriod']

class BrandCreationSerializer(serializers.ModelSerializer):
    colors = ColorSerializer(many=True)
    advanced = AdvancedDataSerializer()

    class Meta:
        model = BrandCreation
        fields = [
            'id', 'logo', 'name', 'titleFont', 'subTitleFont' 
            ,'bodyFont', 'tagLine', 'industry', 'demographic', 
            'psychographic', 'audienceInterest', 'toneOfVoice', 
            'messageStyle', 'visualStyle', 'competitorBrands', 
            'benchmarkBrands', 'brandGuidelineDocs', 'copyrightText', 
            'keywords', 'ctaText', 'currentCampaign', 'visualElements', 
            'colors', 'advanced'
        ]

    def create(self, validated_data):
        colors_data = validated_data.pop('colors', [])
        advanced_data= validated_data.pop('advanced', [])
        
        brand = BrandCreation.objects.create(**validated_data)

        for color_data in colors_data:
            Color.objects.create(brand=brand, **color_data)

        
        Advanced.objects.create(brand=brand, **advanced_data)

        return brand
