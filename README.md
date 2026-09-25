# El factor de equidad actuarial en una fórmula

Material complementario del artículo *Una estimación del Factor de Equidad Actuarial mediante un modelo parsimonioso basado en la función de Weibull* (Marcos González Fernández y Francisco José Sáez Trujillo, Universidad de León), enviado a *Anales del Instituto de Actuarios Españoles*.

## Calculadora interactiva

`index.html` recalcula en el navegador la malla del FdEA por edad de jubilación y años cotizados con el modelo de Weibull. Permite:

- comparar el resultado con las mallas publicadas por el IAE en 2020 y 2025;
- modificar la esperanza de vida a los 65 años y los supuestos macroeconómicos.

No necesita servidor ni instalación.

## Reproducir los resultados del artículo

```bash
cd codigo
pip install numpy pandas scipy matplotlib openpyxl
python fdea_weibull_iae.py
```

El script genera tres salidas:

- `fdea_resultados.xlsx`: mallas coloreadas y métricas;
- `mallas/*.png`: las mallas en imagen;
- `resumen.json`: los resultados en formato de datos.

## Datos

- `datos/per2020_ind_2orden.csv`: tablas PER2020_Ind_2ndo.orden (DGSFP, Resolución de 17 de diciembre de 2020). Contiene la probabilidad de fallecimiento en el año base y el factor de mejora, por sexo.
- Las mallas de FdEA y las tasas de sustitución del IAE están incluidas en el código. Proceden de Devesa Carpio et al. (2020, 2025).
