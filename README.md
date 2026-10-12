# Taller Práctico #2 — EcoMarket: sistema RAG para la atención al cliente

En el [Taller 1](https://github.com/angrinochirandiegofernando-bot/Taller-Pr-ctico-1) el asistente de EcoMarket respondía dos tipos de consulta (estado de un pedido y devolución de un producto) porque los datos se pegaban a mano en el prompt. En este taller el asistente consulta por sí mismo una base de conocimiento de la empresa antes de responder, y reconoce cuándo no tiene información para hacerlo.

| Fase | Dónde está |
|---|---|
| Fase 1: selección de componentes | Este documento |
| Fase 2: base de conocimiento | Este documento y la carpeta [`Datos/`](Datos/) |
| Fase 3: código, experimentos y limitaciones | [`Script/Taller_002.ipynb`](Script/Taller_002.ipynb) |

```
INDEXACIÓN (una vez)
Documentos (PDF, CSV, JSON) -> carga -> segmentación -> embeddings -> ChromaDB

CONSULTA (cada pregunta)
Pregunta -> embedding de la pregunta -> búsqueda por similitud en ChromaDB
         -> fragmentos más parecidos -> prompt (reglas + contexto + pregunta)
         -> modelo de lenguaje -> respuesta con sus fuentes
```

---

## Fase 1: Selección de componentes clave del sistema RAG

### Qué necesita EcoMarket

Las decisiones parten de cuatro características del caso:

- **El idioma es el español**, incluido el español coloquial de un chat ("¿me devuelven la plata?").
- **El volumen de documentos es pequeño.** Políticas, preguntas frecuentes y un catálogo de algunos miles de productos suman decenas de miles de fragmentos, no cientos de millones.
- **El volumen de consultas es alto** (miles al día), así que cada consulta debe ser barata y rápida.
- **Las consultas llevan datos de clientes**, de modo que importa qué información sale de la empresa.

### Modelo de embeddings

El modelo de embeddings determina qué puede encontrar el sistema. Si convierte mal una pregunta en español, el fragmento correcto no llega al modelo de lenguaje y ningún prompt lo arregla después.

| Modelo | Tipo | Español | Dimensiones | Tamaño | Costo | Observación |
|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` (el del ejemplo de clase) | Abierto | Débil: entrenado sobre todo en inglés | 384 | ~90 MB | Gratis | Rápido, pero no es para este idioma |
| `paraphrase-multilingual-MiniLM-L12-v2` | Abierto | Bueno | 384 | ~470 MB | Gratis | Entrenado para similitud entre frases, no para pregunta-documento; entrada máxima de 128 tokens |
| **`intfloat/multilingual-e5-small`** | Abierto | Bueno: cerca de 100 idiomas | 384 | ~470 MB | Gratis | Entrenado para recuperación pregunta-documento; entrada de 512 tokens |
| `BAAI/bge-m3` | Abierto | Muy bueno | 1024 | ~2,2 GB | Gratis, pero pide más memoria | Entrada de 8192 tokens; el más preciso de los abiertos de esta lista |
| `text-embedding-3-small` (OpenAI) | Propietario | Muy bueno | 1536 | Servicio | Del orden de 0,02 USD por millón de tokens | Sin infraestructura propia; los textos salen de la empresa |

**Elección: `intfloat/multilingual-e5-small`, de código abierto y ejecutado en local.**

- **Precisión en español.** La familia E5 se entrena con pares pregunta-documento en muchos idiomas, que es justo la tarea de un RAG. `all-MiniLM-L6-v2` queda descartado por idioma. `paraphrase-multilingual-MiniLM` mide parecido entre frases equivalentes, y una pregunta y su respuesta no son frases equivalentes.
- **Costo.** Para EcoMarket el costo no decide entre abierto y propietario: indexar toda la base de conocimiento son unos pocos millones de tokens, y con un servicio de pago eso cuesta centavos. Lo que sí decide es lo siguiente.
- **Privacidad e independencia.** Con un modelo local, ni los documentos internos ni las preguntas de los clientes salen de la empresa para calcular embeddings, y la búsqueda no depende de que un proveedor esté disponible ni de que mantenga el modelo. Si un proveedor retira su modelo de embeddings hay que volver a indexar todo, porque los vectores de dos modelos distintos no son comparables.
- **Latencia y recursos.** Con 384 dimensiones y cerca de 118 millones de parámetros, calcula el vector de una pregunta en milisegundos en una CPU, sin GPU. `bge-m3` es más preciso, pero ocupa unas cinco veces más memoria y sus vectores son casi tres veces más grandes; para fragmentos cortos como los de este caso, su ventaja de aceptar textos largos no se aprovecha.

**Desventajas de la elección.** Es el modelo pequeño de su familia, así que es menos preciso que `multilingual-e5-large` o `bge-m3`. Exige anteponer `query: ` y `passage: ` a los textos, lo que obliga a escribir un adaptador. Y sus similitudes quedan comprimidas en un rango alto (textos sin relación puntúan cerca de 0,75), lo que hace delicado fijar un umbral de relevancia; el notebook lo calibra con datos.

**Cuándo cambiaría.** Si la evaluación mostrara fallos de recuperación con preguntas reales, el primer paso sería subir a `multilingual-e5-large` o `bge-m3` en un servidor con más memoria. El cambio es una línea de configuración más una reindexación.

### Base de datos vectorial

| | ChromaDB | Pinecone | Weaviate |
|---|---|---|---|
| Modelo | Código abierto; corre dentro de la aplicación o como servidor | Servicio administrado, propietario | Código abierto; se aloja con Docker o Kubernetes, o como servicio en la nube |
| Facilidad de uso | Muy alta: `pip install` y una carpeta en disco | Alta: no hay nada que instalar, pero requiere cuenta, clave y conexión | Media: hay que desplegar y mantener un servidor |
| Escalabilidad | Suficiente para cientos de miles o pocos millones de vectores en una máquina; no es un sistema distribuido | Muy alta y automática | Muy alta, con réplicas y particiones, a cambio de operarlo |
| Costo | Gratis; solo el equipo donde corre | Plan gratuito limitado; después se paga por uso | Gratis si se aloja por cuenta propia (más el costo de operarlo); de pago en la nube |
| Filtros por metadatos | Sí | Sí | Sí |
| Búsqueda híbrida (palabras clave + vectores) | Limitada | Sí | Sí, integrada |
| Dónde quedan los datos | En la infraestructura de EcoMarket | En la nube del proveedor | Donde se despliegue |
| Riesgo principal | Se queda corto si se necesita alta disponibilidad | Dependencia del proveedor y costo creciente con el uso | Complejidad de operación |

FAISS, que usa el ejemplo de clase, no entra en la comparación porque es una biblioteca de búsqueda y no una base de datos: no guarda los textos ni los metadatos, no filtra y la persistencia hay que programarla.

**Elección: ChromaDB en modo persistente.**

- **El tamaño del problema no justifica más.** La base de conocimiento de EcoMarket cabe con holgura en una sola máquina. Pagar o administrar un sistema distribuido para decenas de miles de vectores es complejidad sin beneficio.
- **Los filtros por metadatos son necesarios para este caso.** El asistente los usa para buscar un pedido por su número exacto y para excluir los pedidos de la búsqueda semántica, de modo que nunca muestre el pedido de otro cliente.
- **Costo cero y datos en casa.** No hay cuota mensual ni datos de clientes en la nube de un tercero.
- **Reproducibilidad.** Quien clone el repositorio ejecuta todo sin crear cuentas adicionales.

**Desventajas de la elección.** ChromaDB en este modo es un solo proceso con una carpeta en disco: no tiene réplicas ni alta disponibilidad. Su búsqueda por palabras clave es limitada, lo que importaría si los clientes buscaran mucho por códigos de producto exactos.

**Cuándo cambiaría.** Si el asistente pasara a varios servidores atendiendo en paralelo con exigencia de disponibilidad continua, migraría a Weaviate (si EcoMarket quiere conservar el control de sus datos y aprovechar la búsqueda híbrida) o a Pinecone (si prefiere no administrar infraestructura). Como el código usa la interfaz de almacén vectorial de LangChain, el cambio toca la celda de indexación y no el resto del flujo.

### Cómo afecta cada componente al resultado

| Componente | Si se elige mal |
|---|---|
| Modelo de embeddings | El fragmento con la respuesta no se recupera; el modelo de lenguaje responde con lo que no es o dice que no sabe |
| Base de datos vectorial | La búsqueda es lenta o no permite filtrar; no afecta la calidad de los vectores, sí la latencia, el costo y la privacidad |
| Segmentación (Fase 2) | La respuesta queda partida entre dos fragmentos o diluida en uno demasiado grande |
| Prompt y modelo de lenguaje (Fase 3) | Con el contexto correcto, el modelo igual inventa o ignora las reglas |

---

## Fase 2: Creación de la base de conocimiento

### Documentos identificados

| Documento | Formato | Para qué consultas sirve | Origen |
|---|---|---|---|
| [`politica_devoluciones.pdf`](Datos/politica_devoluciones.pdf) | PDF, 11 secciones | Devoluciones, reembolsos, garantías, plazos | Sintético; contiene las reglas de devolución del prompt del Taller 1, ampliadas |
| [`inventario_productos.csv`](Datos/inventario_productos.csv) | CSV, 16 productos | Precio, existencias, materiales, garantía y certificación de un producto | Sintético |
| [`pedidos.csv`](Datos/pedidos.csv) | CSV, 10 pedidos | Estado de un pedido | Los 10 pedidos reales de Olist usados en el Taller 1 |
| [`preguntas_frecuentes.json`](Datos/preguntas_frecuentes.json) | JSON, 15 preguntas | Envíos, pagos, cuenta, sostenibilidad, contacto | Sintético |

Son tres formatos distintos y cubren el 80% de consultas repetitivas que identificó el Taller 1 (estado del pedido, devoluciones, características del producto), más las preguntas generales que el asistente anterior no podía responder.

El texto de la política y el código que genera el PDF están en el anexo del notebook.

### Estrategias de segmentación

| Estrategia | Cómo corta | Ventaja | Problema |
|---|---|---|---|
| Tamaño fijo | Cada N caracteres | La más simple; fragmentos uniformes | Corta frases y filas por la mitad; ignora la estructura |
| Por párrafos | En cada salto de párrafo | Respeta unidades de sentido | Los fragmentos salen muy desiguales: un párrafo de una línea pierde contexto y uno muy largo mezcla temas |
| Recursiva | Intenta párrafo; si no cabe, frase; si no cabe, palabra | Respeta la estructura hasta donde el tamaño lo permite | Aplicada a ciegas trata un CSV como si fuera prosa |

**Estrategia elegida: segmentación por estructura, distinta para cada tipo de documento.**

- **CSV y JSON: un registro por fragmento.** Un producto, un pedido o una pregunta frecuente ya son unidades completas y cortas (entre 130 y 500 caracteres). Dividir un registro separa el nombre del producto de su precio; juntar varios mezcla productos. Cada fila del CSV se reescribe como una frase ("Producto ECO-007: Botella térmica de acero... Precio: 69900 pesos"), porque un modelo de embeddings entiende lenguaje natural y una fila suelta pierde los nombres de sus columnas. En las preguntas frecuentes, la pregunta y su respuesta van juntas: la pregunta del cliente se parece sobre todo a la pregunta frecuente.
- **PDF: recursiva dentro de cada sección.** Primero se separa por sección, que es la unidad de sentido de una política. Solo las secciones que superan 500 caracteres se dividen de forma recursiva (párrafo, luego frase), con 80 caracteres de solapamiento.
- **Título de la sección en cada fragmento.** Cada fragmento del PDF empieza con el título del documento y de su sección, para que conserve el tema del que habla aunque sea la segunda mitad de una sección.

**Por qué 500 caracteres.** El modelo de embeddings acepta 512 tokens, así que el límite técnico está lejos. El criterio es que un fragmento contenga una sola regla con sus condiciones. Con fragmentos mucho más grandes, el vector promedia varias reglas y deja de parecerse a una pregunta concreta; con fragmentos mucho más pequeños, la regla y su excepción quedan separadas.

**Los metadatos forman parte de la base de conocimiento.** Cada fragmento guarda su archivo de origen, su tipo y un identificador (sección, producto, pedido o pregunta). Con ellos el asistente cita la fuente de cada respuesta, filtra por tipo y encuentra un pedido por su número exacto.

### Evidencia: la segmentación cambia lo que se recupera

El notebook (sección 7) indexa la misma base de tres maneras y mide en cuántas de 14 preguntas el fragmento que contiene la respuesta completa queda entre los primeros resultados:

| Estrategia | Fragmentos | Acierto en el primer resultado | Acierto en los cuatro primeros |
|---|---|---|---|
| Tamaño fijo (300 caracteres) | 40 | 64% | 86% |
| Recursiva genérica (500/80) sobre el texto crudo | 31 | 79% | 100% |
| **Por estructura (la elegida)** | 44 | **93%** | **100%** |

El tamaño fijo falla porque corta en mitad de una fila del inventario: el nombre de un producto queda en un fragmento y su precio en el siguiente, y ningún fragmento contiene la respuesta. La muestra es pequeña, así que las cifras son indicativas.

### Indexación

1. **Carga.** Un cargador por formato convierte cada unidad en un texto con metadatos. En el PDF se extrae el texto por bloques para conservar los párrafos y se normalizan las ligaduras tipográficas ("ﬁ" vuelve a ser "fi"), que de otro modo impedirían reconocer palabras como "confirma".
2. **Segmentación.** Se aplica la estrategia anterior: de 52 unidades salen 54 fragmentos.
3. **Embeddings.** A cada fragmento se le antepone `passage: ` y `multilingual-e5-small` lo convierte en un vector de 384 números, normalizado a longitud 1.
4. **Carga en ChromaDB.** Cada fragmento se guarda con su vector, su texto, sus metadatos y un identificador estable (por ejemplo `ECO-007` o `politica-s4-p1`). ChromaDB construye un índice HNSW con distancia coseno y lo persiste en la carpeta `chroma_db/`.
5. **Actualización.** Como los identificadores son estables, cambiar el inventario de un producto reemplaza un solo registro; no hay que reindexar todo.

En la consulta se hace el recorrido inverso: la pregunta recibe el prefijo `query: `, se convierte en vector con el mismo modelo y ChromaDB devuelve los fragmentos más cercanos.

---

## Fase 3: Integración y ejecución del código

El notebook [`Script/Taller_002.ipynb`](Script/Taller_002.ipynb) implementa el diseño anterior con LangChain y explica cada sección. Parte del asistente del Taller 1: conserva las reglas de sus dos prompts y el mismo cliente compatible con OpenAI, y reemplaza los datos pegados a mano por el contexto recuperado.

| Sección del notebook | Contenido |
|---|---|
| 1 a 5 | Configuración, carga, segmentación, embeddings e indexación |
| 6 | Recuperación: búsqueda semántica, filtro exacto para pedidos y umbral de relevancia calibrado |
| 7 | Experimento 1: efecto de la estrategia de segmentación |
| 8 y 9 | Modelo de lenguaje, prompt y cadena RAG |
| 10 | Pruebas: los dos tipos de consulta del Taller 1 y consultas nuevas |
| 11 | Experimentos 2 a 5: sin RAG y con RAG, prompt estricto y permisivo, número de fragmentos, umbral |
| 12 | Evaluación con 18 preguntas de respuesta conocida |
| 13 y 14 | Limitaciones, suposiciones y conclusiones |
| Anexo | Texto de la política de devoluciones y código que genera su PDF |

### Resultados

El notebook se entrega ejecutado, con `gemini-3.5-flash-lite` como modelo de lenguaje. Lo que muestran sus experimentos:

| Experimento | Qué se cambió | Resultado |
|---|---|---|
| 1. Segmentación | La forma de dividir los documentos | El fragmento correcto queda primero en el 93% de las preguntas con la estrategia elegida y en el 64% con tamaño fijo |
| 2. Sin RAG y con RAG | Se quita la recuperación | Sin RAG el modelo inventó un precio ("$45 USD" en lugar de 189900 pesos), un plazo de reembolso y el estado de un pedido que en realidad estaba cancelado. Con RAG los tres datos coinciden con los documentos |
| 3. Prompt estricto y permisivo | Solo el mensaje de sistema | Ante preguntas sin respuesta en la base, el estricto dice que no tiene la información; el permisivo completa con contenido que ningún documento respalda |
| 4. Número de fragmentos | `k=1` frente a `k=4` | Con un solo fragmento el asistente no pudo responder un caso que combina una regla y su excepción; con cuatro respondió bien |
| 5. Umbral de relevancia | Se desactiva el umbral | El umbral rechaza las preguntas de otro tema sin llamar al modelo; sin umbral y con prompt permisivo, el asistente entrega una receta de cocina |

En la evaluación final el sistema respondió bien 18 de 18 preguntas de respuesta conocida, incluidas 4 en las que lo correcto era decir que no tiene la información ([`Salida/resultados_evaluacion_rag.csv`](Salida/resultados_evaluacion_rag.csv)). Son preguntas propias sobre una base pequeña y se verifican por palabras clave, así que la cifra describe un prototipo y no el desempeño con clientes reales.

### Cómo ejecutarlo

1. Crear un entorno e instalar las dependencias:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
2. Copiar `.env.example` como `.env` y escribir la clave del modelo de lenguaje. Para Gemini, la clave se crea en <https://aistudio.google.com/apikey>. El mismo archivo trae la configuración para usar Ollama u OpenAI.
3. Abrir `Script/Taller_002.ipynb` y ejecutar las celdas en orden. La primera ejecución descarga el modelo de embeddings (unos 470 MB).

### Estructura del repositorio

```
├── README.md                     Fases 1 y 2
├── requirements.txt
├── .env.example                  Plantilla de configuración del modelo de lenguaje
├── Datos/                        Base de conocimiento
│   ├── politica_devoluciones.pdf
│   ├── inventario_productos.csv
│   ├── pedidos.csv
│   └── preguntas_frecuentes.json
├── Script/
│   ├── Taller_002.ipynb          Fase 3: sistema RAG, experimentos y limitaciones
│   └── rag.ipynb                 Ejemplo visto en clase (referencia)
└── Salida/                       Resultados de la evaluación
```

### Limitaciones y suposiciones

Están detalladas en la sección 13 del notebook. Las principales:

- La política, el inventario y las preguntas frecuentes son sintéticos; los pedidos son los 10 reales del Taller 1.
- El equipo de trabajo no tiene GPU y cuenta con 8 GB de RAM, por lo que se usa un modelo de embeddings pequeño y no hay etapa de reordenamiento.
- El modelo de lenguaje se consume por API, así que la pregunta y los fragmentos recuperados salen hacia el proveedor. Los embeddings y la base vectorial se ejecutan en local.
- En producción, el estado de un pedido debe consultarse en la base de datos transaccional, en tiempo real y con el cliente autenticado, no en un índice vectorial.
