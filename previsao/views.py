import joblib
from django.shortcuts import render
from django.http import JsonResponse
import json

# Carregar modelo uma vez
modelo = joblib.load('previsao/models/modelo_treinado_gb_joblib.pkl')

def previsao_view(request):
    if request.method == "POST":
        # Capturar os dados JSON enviados na requisição
        data = json.loads(request.body)  # Agora recebemos diretamente o corpo JSON
        dados = data.get('features')
        
        # Garantir que os dados foram passados corretamente
        if dados:
            # Fazer a previsão
            resultado = modelo.predict([dados])
            
            print(resultado)
            
            if resultado == 0:
                resultado = 'Baixa'
            elif resultado == 1:
                resultado = 'Média'
            else:
                resultado = "Alta"
            # Retornar a previsão como JSON, convertendo o resultado para um tipo serializável
            return JsonResponse({'previsao': resultado})  # Convertendo para int
    
    return render(request, 'previsao.html')
