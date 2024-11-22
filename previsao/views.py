import joblib
from django.shortcuts import render
from django.http import JsonResponse
import json

# Carregar modelo uma vez
modelo = joblib.load('previsao/models/modelo_completo_gb_joblib.pkl')

def previsao_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        dados = data.get('features')
        
        if dados:
            resultado = modelo.predict([dados])
            
            print(resultado)
            
            if resultado == 0:
                resultado = 'Baixa'
            elif resultado == 1:
                resultado = 'Média'
            else:
                resultado = "Alta"
            return JsonResponse({'previsao': resultado})
    
    return render(request, 'previsao.html')
