# Modelos 3D do Blender

Deposite aqui os arquivos dos animais usados para gerar o dataset sintetico.

Extensoes esperadas:

- `.blend`: formato nativo do Blender.
- `.fbx`, `.obj` ou `.glb`: formatos aceitos futuramente pelo pipeline, se necessario.

Os arquivos desta pasta sao ignorados pelo Git por padrao, porque modelos 3D tendem a ser grandes. A estrutura recomendada e separar por classe:

```text
assets/blender_models/
  sapo/
    sapo_01.blend
  peixe/
    peixe_01.blend
  tartaruga/
    tartaruga_01.blend
```

Cada subpasta representa a classe que o modelo de reconhecimento deve aprender.
