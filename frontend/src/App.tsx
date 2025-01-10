import FileUpload from "./components/FileUpload";
import Chat from "./components/Chat";

const App = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center items-center w-full space-y-4">
      <FileUpload />
      <Chat />
    </div>
  );
};

export default App;
