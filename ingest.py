#whenever you want to add additional documents, add the new pdf file in "Data" folder and run this file. 

from rag_app.ingestion import ingest_directory

ingest_directory("data")