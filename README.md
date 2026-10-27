# El factor de equidad actuarial en una fórmula

Material complementario del artículo *El Factor de Equidad Actuarial en una fórmula cerrada: un modelo de Weibull para el sistema español de pensiones* (Marcos González Fernández y Francisco José Sáez Trujillo, Universidad de León), enviado a *Anales del Instituto de Actuarios Españoles*.

## Calculadora interactiva

`index.html` recalcula en el navegador la malla del FdEA por edad de jubilación y años cotizados con el modelo de Weibull. Permite:

- comparar el resultado con las mallas publicadas por el IAE en 2020 y 2025;
- modificar la esperanza de vida a los 65 años y los supuestos macroeconómicos;
- obtener el factor de ajuste de la pensión inicial que mantiene el FdEA cuando cambia la esperanza de vida.

No necesita servidor ni instalación.

## Reproducir los resultados del artículo

```bash
cd codigo
pip install numpy pandas scipy matplotlib openpyxl
python fdea_weibull_iae.py
```

El script genera tres salidas:

- `fdea_resultados.xlsx`: mallas coloreadas, métricas, base reguladora, FdEA medio ponderado, descomposición de 2020 a 2025, contraste de la regla de reescalado y factor de ajuste por longevidad;
- `mallas/*.png`: las mallas en imagen;
- `resumen.json`: todos los resultados en formato de datos.

## Datos

- `datos/per2020_ind_2orden.csv`: tablas PER2020_Ind_2ndo.orden (DGSFP, Resolución de 17 de diciembre de 2020). Contiene la probabilidad de fallecimiento en el año base y el factor de mejora, por sexo.
- Las mallas de FdEA, las tasas de sustitución y los pesos del gasto (MCVL 2018) del IAE están incluidos en el código. Proceden de Devesa Carpio et al. (2020, 2025).
- El IPC de diciembre a diciembre que usa la base reguladora (2018, 2019, 2023 y 2024) procede del INE.

## Versiones

- **v1.2**: base reguladora conforme a la norma: 276 meses entre 322 en 2020 y 300 entre 350 en 2025, con las bases de los dos últimos años por su valor nominal y las anteriores actualizadas con el IPC. Crecimiento real de las bases del 0,22 % en ambos escenarios. Añade la descomposición de 2020 a 2025.
- **v1.1**: añade el FdEA medio ponderado, el contraste de la regla de reescalado y el factor de ajuste por longevidad.
- **v1.0**: versión inicial.
