import uvicorn
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run('src.chatbots:api', host='127.0.0.1', port=6969, reload=True)
