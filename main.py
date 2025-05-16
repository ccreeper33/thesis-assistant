import uvicorn
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.get("api", "host"),
        port=config.getint("api", "port"),
        reload=True
    )
